import itertools
import operator

import PIL.Image

import numpy
import pyopencl

def _read_source(src_file):
    return reduce(operator.concat, open(src_file), '')

class Main:
    HIST_BINS = 256

    def __init__(self):
        ctx   = pyopencl.create_some_context()
        devs  = ctx.devices
        cmdqs = list(pyopencl.CommandQueue(ctx, device=d) for d in devs)

        self._ctx   = ctx
        self._devs  = devs
        self._cmdqs = cmdqs

    def run(self):
        i_cat = PIL.Image.open('data/cat.bmp')

        n_elements = i_cat.size[0] * i_cat.size[1]
        hist_size  = Main.HIST_BINS

        np_img_in = numpy.array(i_cat.getdata(), int)
        np_out_hist = numpy.zeros(hist_size, int)

        print 'cat size: {}x{} = {} pixels'.format(
                i_cat.size[0], i_cat.size[1],
                len(np_img_in))

        buf_in_img = pyopencl.Buffer(
                self._ctx,
                pyopencl.mem_flags.READ_ONLY |
                pyopencl.mem_flags.COPY_HOST_PTR,
                hostbuf=np_img_in)
        buf_out_hist = pyopencl.Buffer(
                self._ctx,
                pyopencl.mem_flags.WRITE_ONLY,
                np_out_hist.nbytes)

        pyopencl.enqueue_fill_buffer(
                self._cmdqs[0],
                buf_out_hist,
                numpy.int32(0),
                0,
                np_out_hist.nbytes)

        program = pyopencl.Program(
                self._ctx,
                _read_source('histogram.cl')).build(devices=[self._devs[0]])

        program.histogram(
                self._cmdqs[0],
                (1024,),
                (64,),
                buf_in_img,
                numpy.int32(n_elements),
                buf_out_hist)
        #print np_out_hist

        pyopencl.enqueue_copy(
                self._cmdqs[0],
                np_out_hist,
                buf_out_hist)

        print "reference histogram:"
        print i_cat.histogram()

        print "OpenCL computed histogram:"
        print np_out_hist

        #print '\n'.join(map(str, np_out_hist))

        #print len(list(i_cat.getdata()))
        #print ', '.join(map(str,i_cat.getdata()))
        #print len(i_cat.histogram())
        #print numpy.array(i_cat.histogram(), int)
        #print '\n'.join(map(str,i_cat.histogram()))

if __name__ == '__main__':
    Main().run()
