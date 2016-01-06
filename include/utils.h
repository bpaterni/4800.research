#ifndef __CSBS_UTILS_H
#define __CSBS_UTILS_H

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

#endif  /* __CSBS_UTILS_H */
