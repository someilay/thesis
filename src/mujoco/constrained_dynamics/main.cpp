#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <cstring>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "GLFW/glfw3.h"
#include "mujoco/mujoco.h"


using std::printf;
using std::string;
using std::cout;
using Mat4d = Eigen::Matrix4d;
using Vec4d = Eigen::Vector4d;
using Vec3d = Eigen::Vector3d;
using Vec2d = Eigen::Vector2d;

template<typename T, int R, int C>
using Mat = Eigen::Matrix<T, R, C>;

template<typename T, int S>
using Vec = Eigen::Vector<T, S>;


#define G 10e0
#define M 1e0
#define L 1e0
#define P_MAT Mat<double, 2, 4> {{0, 0, 1, 0}, {0, 0, 0, 1}}
#define K_D (10 * Eigen::MatrixXd::Identity(2, 2))
#define K_P (5 * Eigen::MatrixXd::Identity(2, 2))
#define TIMES_SLOWER 1


Mat<double, 2, 4> jacConstraint(Vec4d &q) {
    double x_1 = q[0], y_1 = q[1], x_2 = q[2], y_2 = q[3];
    return 2 * Mat<double, 2, 4>{
            {x_1,       y_1,       0,         0},
            {x_1 - x_2, y_1 - y_2, x_2 - x_1, y_2 - y_1}
    };
}


Mat<double, 2, 4> dotJacConstraint(Vec4d &dot_q) {
    double dot_x_1 = dot_q[0], dot_y_1 = dot_q[1], dot_x_2 = dot_q[2], dot_y_2 = dot_q[3];
    return 2 * Mat<double, 2, 4>{
            {dot_x_1,           dot_y_1,           0,                 0},
            {dot_x_1 - dot_x_2, dot_y_1 - dot_y_2, dot_x_2 - dot_x_1, dot_y_2 - dot_y_1}
    };
}


Vec4d hVal() {
    return {0, M * G, 0, M * G};
}


Mat4d massMatrix() {
    return M * Eigen::MatrixXd::Identity(4, 4);
}


Vec2d ddotQDesired(double t) {
    return {0, -L * sin(t)};
}


Vec2d dotQTilda(Vec4d &dot_q, double t) {
    double dot_x_2 = dot_q[2], dot_y_2 = dot_q[3];
    return {dot_x_2, dot_y_2 - L * cos(t)};
}


Vec2d qTilda(Vec4d &q, double t) {
    double x_2 = q[2], y_2 = q[3];
    return {x_2 - L, y_2 - L * (sin(t) + 0.2)};
}


Mat<double, 8, 8> fullMatrix(Vec4d &q) {
    Mat<double, 8, 8> res = Eigen::MatrixXd::Zero(8, 8);

    res.block(0, 0, 4, 4) = massMatrix();
    res.block(0, 4, 4, 2) = jacConstraint(q).transpose();
    res.block(4, 0, 2, 4) = jacConstraint(q);
    res.block(0, 6, 4, 2) = P_MAT.transpose();
    res.block(6, 0, 2, 4) = P_MAT;

    return res;
}


Vec<double, 8> getRightSide(Vec4d &q, Vec4d &dot_q, double t) {
    Vec4d h = hVal();
    Mat<double, 2, 4> dot_jac = dotJacConstraint(dot_q);
    Vec2d dump = ddotQDesired(t) - K_D * dotQTilda(dot_q, t) - K_P * qTilda(q, t);

    Vec<double, 8> res;
    res.head(4) = -h;
    res.segment(4, 2) = -dot_jac * dot_q;
    res.tail(2) = dump;
    return res;
}


Vec2d getControl(Vec4d &q, Vec4d &dot_q, double t) {
    Vec<double, 8> right_side = getRightSide(q, dot_q, t);
    Mat<double, 8, 8> n_mat = fullMatrix(q);
    return -(n_mat.inverse() * right_side).tail(2);
}


void setSplitQ(Vec3d* q1, Vec3d* q2, mjData* d) {
    double* xmat = d->xmat;
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> first_t(xmat + 1 * 9);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> second_t(xmat + 2 * 9);

    *q1 = first_t * Vec3d{0, L, 0};
    *q2 = *q1 + second_t * Vec3d{0, L, 0};
}


Vec4d getQ(mjData* d) {
    Vec3d first_pos, second_pos;
    setSplitQ(&first_pos, &second_pos, d);

    return {first_pos[1], first_pos[2], second_pos[1], second_pos[2]};
}


Vec4d getQDot(mjData* d) {
    double* qvel = d->qvel;
    Vec3d first_pos, second_pos;
    setSplitQ(&first_pos, &second_pos, d);

    Vec3d vel_1 = (Vec3d{qvel[0], 0, 0}).cross(first_pos);
    Vec3d vel_2 = (Vec3d{qvel[0] + qvel[1], 0, 0}).cross(second_pos - first_pos) + vel_1;

    return {vel_1[1], vel_1[2], vel_2[1], vel_2[2]};
}


bool exists(const string &name) {
    struct stat buffer{};
    int fd = stat(name.c_str(), &buffer);
    if (fd > -1) close(fd);
    return fd == 0;
}


int main() {
    // MuJoCo data structures
    char error[1000];
    memset(error, 0, 1000 * sizeof(char));
    string path = "double_pendulum.xml";
    string alt_path = "../" + path;

    mjModel* m = nullptr;
    mjData* d;
    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;

    if (exists(path)) {
        m = mj_loadXML(path.c_str(), nullptr, error, 1000);
    }
    if (exists(alt_path) && !m) {
        m = mj_loadXML(alt_path.c_str(), nullptr, error, 1000);
    }
    if (error[0]) {
        printf("Error occurred: %s\n", error);
        return 1;
    }
    if (!m) {
        printf("Neither %s, neither %s has been found!\n", path.c_str(), alt_path.c_str());
        return 1;
    }
    // init data
    d = mj_makeData(m);

    // init GLFW, create window, make OpenGL context current, request v-sync
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjv_defaultScene(&scn);

    cam.distance = 4;
    cam.elevation = 0;
    cam.azimuth = -180;

    printf("M = %f, L = %f, G = %f\n", M, L, G);
    printf("nbody = %d\n", m->nbody);
    printf("nq = %d\n", m->nq);
    printf("nv = %d\n", m->nv);
    printf(
            "cam.elevation = %f\n"
            "cam.azimuth = %f\n"
            "cam.distance = %f\n",
            cam.elevation,
            cam.azimuth,
            cam.distance
    );

    // create scene and context
    // opt.frame = mjFRAME_BODY; //enable body frames
    mjv_makeScene(m, &scn, 1000);
    mjr_makeContext(m, &con, mjFONTSCALE_100);

    // run main loop, target real-time simulation and 60 fps rendering
    Vec4d prev_q = Vec4d::Zero();
    while (!glfwWindowShouldClose(window)) {
        mjtNum sim_start = d->time;
        while (d->time - sim_start < 1.0 / (60.0 * TIMES_SLOWER)) {
            auto q = getQ(d);
            auto dot_q = getQDot(d);
            auto ctrl = getControl(q, dot_q, d->time);
            d->xfrc_applied[2 * 6 + 1] = ctrl[0];
            d->xfrc_applied[2 * 6 + 2] = ctrl[1];

            mj_step(m, d);
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        // cam.azimuth += 0.25;
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    free(m);
    free(d);
    return 0;
}
