import argparse
import itertools
import multiprocessing
import operator
import os
import random
import sys
import time

import numpy
import pyopencl

class ToIntAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, int(values))

def _addcomp(x):
    return x[0] + x[1]

class VecAdd_Serial:
    def do_vadd(self, A, B):
        return list(itertools.imap(lambda a, b: a+b, A, B))

class VecAdd_Coarse:
    def do_vadd(self, AB):
        return list(
            multiprocessing.Pool().map(
                _addcomp,
                AB,
                chunksize=len(AB)/multiprocessing.cpu_count()))

class VecAdd_CL:
    def __init__(self, A, B):
        ctx     = pyopencl.create_some_context()
        devs    = ctx.devices
        program = pyopencl.Program(
                ctx,
                self._read_source('vecadd.cl')).build(devices=devs)
        cmd_qs  = list(pyopencl.CommandQueue(ctx, device=dev) for dev in devs)

        npA = numpy.array(A, int)
        npB = numpy.array(B, int)
        npC = numpy.array(A, int)

        bufA = pyopencl.Buffer(
                ctx,
                pyopencl.mem_flags.READ_ONLY |
                pyopencl.mem_flags.COPY_HOST_PTR,
                hostbuf=npA)
        bufB = pyopencl.Buffer(
                ctx,
                pyopencl.mem_flags.READ_ONLY |
                pyopencl.mem_flags.COPY_HOST_PTR,
                hostbuf=npB)
        bufC = pyopencl.Buffer(
                ctx,
                pyopencl.mem_flags.WRITE_ONLY,
                npA.nbytes)

        self._ctx = ctx
        self._devs    = devs
        self._program = program
        self._cmd_qs  = cmd_qs
        self._npas    = [npA, npB, npC]
        self._bufs    = [bufA, bufB, bufC]

    def _read_source(self, src_file):
        return reduce(operator.concat, open(src_file), '')

    def do_vadd(self):
        self._program.vecadd(
                self._cmd_qs[0],
                self._npas[0].shape,
                None,
                *self._bufs)
        pyopencl.enqueue_copy(
                self._cmd_qs[0],
                self._npas[2],
                self._bufs[2])

        return self._npas[2]

class Main:
    DESC = 'Runtime system for OpenCL programs'
    NP   = multiprocessing.cpu_count()

    VAL_RANGE_MAX = sys.maxint >> 31
    VAL_RANGE_MIN = -VAL_RANGE_MAX

    def __init__(self):
        argparser = argparse.ArgumentParser(Main.DESC)

        argparser.add_argument(
                '-l',
                nargs='?',
                default=10,
                help='length of list to process',
                action=ToIntAction)

        self._options = vars(argparser.parse_args())

        self._length = self._options['l']

        self._A = list(
                itertools.takewhile(
                    lambda x: x < self._length,
                    itertools.count()))
        self._B = list(self._A)

    def run(self):
        va_serial = VecAdd_Serial()
        start = time.time()
        C = va_serial.do_vadd(self._A, self._B)
        end   = time.time()

        print 'VecAdd_Serial: {}'.format(end-start)

        AB = zip(self._A, self._B)
        va_coarse = VecAdd_Coarse()
        start = time.time()
        C = va_coarse.do_vadd(AB)
        end   = time.time()

        print 'VecAdd_Coarse: {}'.format(end-start)

        va_cl = VecAdd_CL(self._A, self._B)
        start = time.time()
        va_cl.do_vadd()
        end   = time.time()

        print 'VecAdd_CL: {}'.format(end - start)

if __name__ == '__main__':
    Main().run()
