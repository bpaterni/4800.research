#ifndef __CSBS_UTILS_H
#define __CSBS_UTILS_H

#include <CL/cl.h>

#define PARSE_OPTS_WITH_ENTRIES(desc, entries)                           \
    GError *error = NULL;                                                \
    GOptionContext *opt_ctx;                                             \
                                                                         \
    opt_ctx = g_option_context_new(desc);                                \
    g_option_context_add_main_entries(opt_ctx, entries, NULL);           \
    if( !g_option_context_parse(opt_ctx, &argc, &argv, &error) ) {       \
        g_print("option parsing failed: %s\n", error->message);          \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
    g_option_context_free(opt_ctx);

void check(cl_int),
     print_cl_compiler_error(cl_program, cl_device_id);

#endif  /* __CSBS_UTILS_H */
