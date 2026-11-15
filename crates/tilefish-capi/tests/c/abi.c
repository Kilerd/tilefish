/*
 * The C ABI from C99, with no GPU: the header alone (it compiles as C99 and C++17), the layout of its structs, the ABI version, the statuses of
 * a failure and the thread's last error, and loading the committed manifests and plans with their pins (and refusing wrong ones).
 *
 *     tilefish_c_abi <manifest dir> <manifest pin> <plan dir> <plan pin> <plan set dir> <plan set pin>
 *
 * (ctest passes the committed fixtures: crates/tilefish/tests/fixtures/sm_103, python/tests/golden/plans/block and .../carried. The golden plans
 * hold fake cubins, which no driver loads: they are read here, never instantiated.)
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tilefish.h"
#include "tilefish.h" /* twice: the include guard holds */

/* the layout tests/abi.rs checks on the Rust side (a negative array size does not compile) */
typedef char tensor_size[(sizeof(tilefish_tensor) == 56) ? 1 : -1];
typedef char tensor_device[(offsetof(tilefish_tensor, device) == 16) ? 1 : -1];
typedef char tensor_rank[(offsetof(tilefish_tensor, rank) == 28) ? 1 : -1];
typedef char tensor_shape[(offsetof(tilefish_tensor, shape) == 32) ? 1 : -1];
typedef char tensor_offset[(offsetof(tilefish_tensor, offset) == 48) ? 1 : -1];
typedef char named_size[(sizeof(tilefish_named_tensor) == 64) ? 1 : -1];
typedef char tap_size[(sizeof(tilefish_tap) == 72) ? 1 : -1];
typedef char devscalar_size[(sizeof(tilefish_devscalar) == 32) ? 1 : -1];
typedef char devscalar_int[(offsetof(tilefish_devscalar, int_value) == 16) ? 1 : -1];
typedef char bucket_size[(sizeof(tilefish_bucket) == 16) ? 1 : -1];
typedef char given_size[(sizeof(tilefish_given) == 16) ? 1 : -1];
typedef char file_size[(sizeof(tilefish_file) == 24) ? 1 : -1];
typedef char error_size[(sizeof(tilefish_error_info) == 40) ? 1 : -1];
typedef char buffer_size[(sizeof(tilefish_buffer_info) == 32) ? 1 : -1];

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                   \
        }                                                                 \
    } while (0)

static tilefish_error_info last(void) {
    tilefish_error_info info;
    tilefish_last_error(&info);
    return info;
}

static void expect_status(tilefish_status got, tilefish_status want, const char *what) {
    if (got != want) {
        tilefish_error_info info = last();
        fprintf(stderr, "%s: got %s, want %s (%s)\n", what, tilefish_status_name(got), tilefish_status_name(want), info.message);
        failures++;
    }
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <manifest dir> <manifest pin> <plan dir> <plan pin> <plan set dir> <plan set pin>\n", argv[0]);
        return 2;
    }
    const char *manifest_dir = argv[1], *manifest_pin = argv[2], *plan_dir = argv[3], *plan_pin = argv[4];
    const char *set_dir = argv[5], *set_pin = argv[6];

    /* versions */
    CHECK(tilefish_abi_compatible());
    CHECK((tilefish_abi_version() >> 16) == TILEFISH_ABI_MAJOR);
    CHECK(strlen(tilefish_version()) > 0);
    CHECK(tilefish_manifest_schema() == 3);
    CHECK(strcmp(tilefish_plan_schema(), "tilefish.plan.v1") == 0);
    CHECK(strcmp(tilefish_status_name(TILEFISH_ERROR_PIN_MISMATCH), "TILEFISH_ERROR_PIN_MISMATCH") == 0);

    /* a manifest, with its pin, a wrong pin, a bad pin, trusting the directory, a missing directory */
    tilefish_manifest *m = NULL;
    expect_status(tilefish_manifest_load(manifest_dir, manifest_pin, &m), TILEFISH_OK, "load with the pin");
    CHECK(m != NULL && strcmp(tilefish_manifest_digest(m), manifest_pin) == 0);
    tilefish_manifest_destroy(m);
    char wrong[65];
    memset(wrong, '0', 64);
    wrong[64] = '\0';
    m = (tilefish_manifest *)&failures; /* a failure leaves NULL */
    expect_status(tilefish_manifest_load(manifest_dir, wrong, &m), TILEFISH_ERROR_PIN_MISMATCH, "a wrong pin");
    CHECK(m == NULL);
    tilefish_error_info info = last();
    CHECK(info.status == TILEFISH_ERROR_PIN_MISMATCH && strstr(info.message, manifest_pin) != NULL);
    CHECK(info.precondition == TILEFISH_PRECONDITION_NONE && info.argument == NULL);
    expect_status(tilefish_manifest_load(manifest_dir, "abc", &m), TILEFISH_ERROR_BAD_PIN, "a bad pin");
    expect_status(tilefish_manifest_load("/no/such/tilefish/directory", manifest_pin, &m), TILEFISH_ERROR_IO, "no directory");
    expect_status(tilefish_manifest_load_trusting_directory(manifest_dir, &m), TILEFISH_OK, "trusting the directory");
    tilefish_manifest_destroy(m);
    expect_status(tilefish_manifest_load(NULL, manifest_pin, &m), TILEFISH_ERROR_INVALID_ARGUMENT, "a NULL directory");
    expect_status(tilefish_manifest_load(manifest_dir, manifest_pin, NULL), TILEFISH_ERROR_INVALID_ARGUMENT, "a NULL out");

    /* a plan and what it says of itself */
    tilefish_plan *p = NULL;
    expect_status(tilefish_plan_load(plan_dir, plan_pin, &p), TILEFISH_OK, "a plan with its pin");
    CHECK(strcmp(tilefish_plan_digest(p), plan_pin) == 0);
    CHECK(strcmp(tilefish_plan_graph(p), "block") == 0);
    CHECK(tilefish_plan_output_count(p) == 1 && strcmp(tilefish_plan_output(p, 0), "out0") == 0);
    int devscalars = 0;
    for (size_t i = 0; i < tilefish_plan_buffer_count(p); i++) {
        tilefish_buffer_info b;
        expect_status(tilefish_plan_buffer(p, i, &b), TILEFISH_OK, "a buffer");
        devscalars += b.role == TILEFISH_ROLE_DEVSCALAR;
    }
    CHECK(devscalars == 1);
    const char *dim = NULL;
    const uint64_t *values = NULL;
    size_t n_values = 0;
    expect_status(tilefish_plan_dim(p, 0, &dim, &values, &n_values), TILEFISH_OK, "a dimension");
    CHECK(strcmp(dim, "M") == 0 && n_values == 3 && values[0] == 32);
    tilefish_plan_destroy(p);
    expect_status(tilefish_plan_load(plan_dir, wrong, &p), TILEFISH_ERROR_PLAN_PIN_MISMATCH, "a plan with a wrong pin");

    /* a plan set */
    tilefish_plan_set *set = NULL;
    expect_status(tilefish_plan_set_load(set_dir, set_pin, &set), TILEFISH_OK, "a plan set with its pin");
    CHECK(tilefish_plan_set_name_count(set) > 1);
    const char *first = tilefish_plan_set_name(set, 0);
    expect_status(tilefish_plan_set_plan(set, first, &p), TILEFISH_OK, "a plan of the set");
    CHECK(strcmp(tilefish_plan_graph(p), first) == 0);
    tilefish_plan_destroy(p);
    tilefish_plan_set_destroy(set);

    /* a context: refused without a driver */
    tilefish_context *ctx = NULL;
    tilefish_status status = tilefish_context_primary(0, &ctx);
    if (!tilefish_driver_available()) {
        expect_status(status, TILEFISH_ERROR_NO_DRIVER, "a context without a driver");
    } else if (status == TILEFISH_OK) {
        uint32_t arch = 0;
        expect_status(tilefish_context_arch(ctx, &arch), TILEFISH_OK, "the architecture");
        CHECK(arch >= 70);
        CHECK(tilefish_context_handle(ctx) != NULL && tilefish_context_device(ctx) == 0);
    }
    tilefish_context_destroy(ctx);

    if (failures) {
        fprintf(stderr, "%d checks failed\n", failures);
        return 1;
    }
    printf("tilefish C ABI: every check passed (ABI %u.%u, tilefish %s)\n", tilefish_abi_version() >> 16, tilefish_abi_version() & 0xffffu,
           tilefish_version());
    return 0;
}
