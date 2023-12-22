#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <cstring>
#include <cmath>

#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "GLFW/glfw3.h"
#include "mujoco/mujoco.h"

namespace pin = pinocchio;

using std::printf;
using std::string;
using std::cout;
using Mat4d = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
using MatXd = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
using Vec4d = Eigen::Vector4d;
using Vec3d = Eigen::Vector3d;
using Vec2d = Eigen::Vector2d;
using VecXd = Eigen::VectorXd;

template<typename T, int R, int C>
using Mat = Eigen::Matrix<T, R, C>;

template<typename T, int S>
using Vec = Eigen::Vector<T, S>;


#define TIMES_SLOWER 3
#define PAUSED false
#define KP 100
#define KD 20


void printVector(double* arr, int len) {
    if (len == 0) {
        printf("[]\n");
        return;
    }

    printf("[");
    for (int i = 0; i < len - 1; ++i) {
        printf("%.4f, ", arr[i]);
    }
    printf("%.4f]\n", arr[len - 1]);
}


bool exists(const string &name) {
    struct stat buffer{};
    int fd = stat(name.c_str(), &buffer);
    if (fd > -1) close(fd);
    return fd == 0;
}


MatXd getMassMatrix(pin::Model &m, pin::Data &d, VecXd &q) {
    MatXd massMatrix = pin::crba(m, d, q);
    massMatrix.triangularView<Eigen::StrictlyLower>() = massMatrix.transpose().triangularView<Eigen::StrictlyLower>();
    return massMatrix;
}


VecXd se3error(MatXd rot, MatXd dRot) {
    MatXd errorRot = rot.transpose() * dRot;
    MatXd errorLog = errorRot.log();
    VecXd error(3); error << errorLog(2, 1), errorLog(0, 2), errorLog(1, 0); 
    return error;
}


VecXd posError(VecXd pos, MatXd rot, VecXd dPos, MatXd dRot) {
    VecXd totalError(6);
    totalError.segment(0, 3) = dPos - pos;
    totalError.segment(3, 3) = se3error(rot, dRot);
    return totalError;
}


VecXd getConstraintForces(pin::Model &m,
                          pin::Data &d,
                          VecXd &q,
                          VecXd &v,
                          VecXd &f,
                          MatXd& dRot,
                          VecXd& dPos,
                          VecXd& dVel,
                          VecXd& dAcc,
                          pin::JointIndex jIdx) {
    // Get mass matrix
    MatXd massMatrix = getMassMatrix(m, d, q);

    // Compute jacobians
    pin::computeJointJacobians(m, d, q);
    pin::computeJointJacobiansTimeVariation(m, d, q, v);

    // Get jacobians
    MatXd jac(6, m.nv);
    MatXd dJac(6, m.nv);
    MatXd jacLO(6, m.nv);
    MatXd dJacLO(6, m.nv);

    pin::getJointJacobian(m, d, jIdx, pin::LOCAL_WORLD_ALIGNED, jac);
    pin::getJointJacobianTimeVariation(m, d, jIdx, pin::LOCAL_WORLD_ALIGNED, dJac);
    pin::getJointJacobian(m, d, jIdx, pin::LOCAL, jacLO);
    pin::getJointJacobianTimeVariation(m, d, jIdx, pin::LOCAL, dJacLO);

    jac.block(3, 0, 3, m.nv) = jacLO.block(3, 0, 3, m.nv);
    dJac.block(3, 0, 3, m.nv) = dJacLO.block(3, 0, 3, m.nv);

    // Compute M^-1, M^(1 / 2) and M^(-1 / 2)
    MatXd invM = massMatrix.inverse();
    MatXd sqrtM = massMatrix.sqrt();
    MatXd invSqrtM = sqrtM.inverse();

    // Compute AM^-1 and [AM^(-1 / 2)]^+
    MatXd aM = jac * invM;
    MatXd pAiSqrtM = (jac * invSqrtM).completeOrthogonalDecomposition().pseudoInverse();

    // Compute current position & orientation
    VecXd pos = d.oMi[jIdx].translation();
    MatXd rot = d.oMi[jIdx].rotation();

    // Compute error
    VecXd pE = posError(pos, rot, dPos, dRot);
    VecXd dpE = dVel - jac * v;

    // Compute b with KD control
    VecXd b = dAcc + KP * pE + KD * dpE - dJac * v;

    // Compute Q_c
    VecXd fC = sqrtM * pAiSqrtM * (b - aM * f);
    return fC;
}


int main(int argc, char **argv) {
    // MuJoCo data structures
    char error[1000];
    memset(error, 0, 1000 * sizeof(char));
    // string mjModelPath = "../kuka_iiwa_14/scene.xml";
    string mjModelPath = argv[1];
    string pinURDFPath = argv[2];

    pinocchio::Model p_m;
    pinocchio::Data p_d;

    mjModel* m;
    mjData* d;
    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;

    if (!exists(pinURDFPath)) {
        printf("%s not has been found!\n", pinURDFPath.c_str());
        return 1;
    }
    if (!exists(mjModelPath)) {
        printf("%s not has been found!\n", mjModelPath.c_str());
        return 1;
    }

    // init pinocchio
    pinocchio::urdf::buildModel(pinURDFPath, p_m);
    p_d = pinocchio::Data(p_m);

    // init mujoco data
    m = mj_loadXML(mjModelPath.c_str(), nullptr, error, 1000);
    d = mj_makeData(m);
    if (error[0]) {
        printf("Error occurred: %s\n", error);
        return 1;
    }

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

    cam.distance = 1;
    cam.elevation = -30;
    cam.azimuth = -170;

    printf("nbody = %d\n", m->nbody);
    printf("nq = %d\n", m->nq);
    printf("nv = %d\n", m->nv);
    printf("njt = %d\n", m->njnt);
    printf(
            "cam.elevation = %f\n"
            "cam.azimuth = %f\n"
            "cam.distance = %f\n",
            cam.elevation,
            cam.azimuth,
            cam.distance
    );

    VecXd q(m->nv);
    VecXd qVel(m->nv);
    VecXd qForce(m->nv);
    VecXd qC(m->nv);

    // create scene and context
    mjv_makeScene(m, &scn, 1000);
    mjr_makeContext(m, &con, mjFONTSCALE_100);

    // init simulation
    mj_checkPos(m, d);
    mj_checkVel(m, d);
    mj_forward(m, d);
    mj_checkAcc(m, d);

    // set necessary values
    pinocchio::JointIndex joint6Idx = p_m.getJointId("joint6");
    printf("joint6Idx = %ld\n", joint6Idx);

    VecXd dPos(3); dPos << 0, 0.3, 0.5;
    MatXd dRot = MatXd::Identity(3, 3);
    VecXd dVel(6); dVel << 0, 0, 0, 0, 0, 0;
    VecXd dAcc(6); dAcc << 0, 0, 0, 0, 0, 0;

    while (!glfwWindowShouldClose(window)) {
        mjtNum sim_start = d->time;
        while (d->time - sim_start < 1.0 / (60.0 * TIMES_SLOWER) && !PAUSED) {
            // copy state from mujoco
            memcpy(q.data(), d->qpos, m->nq * sizeof(mjtNum));
            memcpy(qVel.data(), d->qvel, m->nv * sizeof(mjtNum));
            memcpy(qForce.data(), d->qfrc_bias, m->nv * sizeof(mjtNum));

            // compute constraint forces
            qC = getConstraintForces(p_m, p_d, q, qVel, qForce, dRot, dPos, dVel, dAcc, joint6Idx);
            memcpy(d->ctrl, qC.data(), m->nv * sizeof(mjtNum));

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
