// Standalone C++ replay: reads a step CSV and runs the core with no Python.
// Proves libidr runs identically off the app/edge path.
//   cols: dt,gyro_z,v_meas,v_sigma,a_lat,mask_gnss,e_gnss,n_gnss
// v_meas<0 => no speed update this step; mask_gnss=0 => apply GNSS pos update.
#include "../core/idr.h"
#include <cstdio>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: replay steps.csv\n"); return 1; }
    FILE* fp = std::fopen(argv[1], "r");
    if (!fp) { std::perror("open"); return 1; }
    char line[512];
    if (!std::fgets(line, sizeof line, fp)) return 1;   // header
    idr_filter* f = idr_create();
    double e0, n0, psi0, v0;
    // first data row seeds the state
    if (std::fgets(line, sizeof line, fp)) {
        double dt, gz, vm, vs, al, mask, eg, ng;
        std::sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &dt,&gz,&vm,&vs,&al,&mask,&eg,&ng);
        e0 = eg; n0 = ng; psi0 = 0; v0 = vm > 0 ? vm : 0;
        idr_init(f, e0, n0, psi0, v0);
    }
    std::rewind(fp); std::fgets(line, sizeof line, fp);   // skip header again
    double st[5];
    long steps = 0;
    while (std::fgets(line, sizeof line, fp)) {
        double dt, gz, vm, vs, al, mask, eg, ng;
        if (std::sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &dt,&gz,&vm,&vs,&al,&mask,&eg,&ng) != 8) continue;
        idr_predict(f, dt, gz);
        if (vm >= 0) idr_update_speed(f, vm, vs);
        double psidot = gz;   // approx; core gates internally
        idr_update_curvature(f, al, psidot, 2.0);
        if (mask == 0) idr_update_gnss_pos(f, eg, ng, 3.0);
        steps++;
    }
    idr_get_state(f, st);
    std::printf("steps=%ld  e=%.3f n=%.3f psi=%.4f v=%.3f bg=%.6f\n",
                steps, st[0], st[1], st[2], st[3], st[4]);
    idr_destroy(f);
    std::fclose(fp);
    return 0;
}
