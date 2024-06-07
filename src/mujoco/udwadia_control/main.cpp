#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "GLFW/glfw3.h"
#include "inicpp.h"
#include "mujoco/mujoco.h"

namespace pin = pinocchio;

using std::cout;
using std::printf;
using std::string;
using Mat4d = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
using Mat3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using MatXd = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
using Vec4d = Eigen::Vector4d;
using Vec3d = Eigen::Vector3d;
using Vec2d = Eigen::Vector2d;
using VecXd = Eigen::VectorXd;

template <typename T, int R, int C>
using Mat = Eigen::Matrix<T, R, C>;

template <typename T, int S>
using Vec = Eigen::Vector<T, S>;

#define DEFAULT_CONFIG_PATH "config.ini"

// Constants
float TIMES_SLOWER = 3;
bool PAUSED = false;
float MAX_DURATION = INFINITY;
double KP = 10;
double KD = 2;
double TELEMETRY_PRINT_DELTA = 0.5;
bool SHOW_ERROR = false;
bool SHOW_QC = false;
bool SHOW_Q = false;
bool SHOW_Q_VEL = false;
bool SHOW_Q_BIAS = false;
double CAM_DISTANCE = 4;
double CAM_ELEVATION = -30;
double CAM_AZIMUTH = 10;

string MUJOCO_XML_PATH = "scene.xml";
string URDF_PATH = "urdf_example.urdf";
string PIN_OBJ_NAME = "gripper-body_5-joint";
string MUJ_OBJ_NAME = "gripper";

VecXd dPos(3);
MatXd dRot = MatXd::Identity(3, 3);

void printVector(double *arr, int len, bool newLine = true) {
    if (len == 0) {
        printf("[]\n");
        return;
    }

    printf("[");
    for (int i = 0; i < len - 1; ++i) {
        printf("%.4f, ", arr[i]);
    }
    printf("%.4f]", arr[len - 1]);

    if (newLine)
        printf("\n");
}

bool exists(const string &name) {
    struct stat buffer {
    };
    int fd = stat(name.c_str(), &buffer);
    if (fd > -1)
        close(fd);
    return fd == 0;
}

template <typename T>
T tryToReadField(const inicpp::config &content, string collection, const string field, T defaultValue) {
    try {
        return content[collection][field].get<T>();
    } catch (inicpp::not_found_exception &e) {
        cout << e.what() << ", using default value: " << defaultValue << "\n";
    }
    return defaultValue;
}

void readIniConfig(string config_path) {
    // Init dPos
    dPos << 0, 0.1, 1.1;

    if (!exists(config_path)) {
        printf("%s not found, using default values!\n", config_path.c_str());
        return;
    }
    inicpp::config content = inicpp::parser::load_file(config_path);

    MUJOCO_XML_PATH = tryToReadField<inicpp::string_ini_t>(content, "Simulation", "mujoco_xml_path", MUJOCO_XML_PATH);
    URDF_PATH = tryToReadField<inicpp::string_ini_t>(content, "Simulation", "urdf_path", URDF_PATH);
    TIMES_SLOWER = tryToReadField<inicpp::float_ini_t>(content, "Simulation", "times_slower", TIMES_SLOWER);
    PAUSED = tryToReadField<inicpp::boolean_ini_t>(content, "Simulation", "paused", PAUSED);
    MAX_DURATION = tryToReadField<inicpp::float_ini_t>(content, "Simulation", "max_duration", MAX_DURATION);
    TELEMETRY_PRINT_DELTA =
        tryToReadField<inicpp::float_ini_t>(content, "Simulation", "telemetry_print_delta", TELEMETRY_PRINT_DELTA);
    PIN_OBJ_NAME = tryToReadField<inicpp::string_ini_t>(content, "Simulation", "pinocchio_obj_name", PIN_OBJ_NAME);
    MUJ_OBJ_NAME = tryToReadField<inicpp::string_ini_t>(content, "Simulation", "mujoco_obj_name", MUJ_OBJ_NAME);

    SHOW_ERROR = tryToReadField<inicpp::boolean_ini_t>(content, "Simulation", "show_error", SHOW_ERROR);
    SHOW_Q = tryToReadField<inicpp::boolean_ini_t>(content, "Simulation", "show_q", SHOW_Q);
    SHOW_Q_VEL = tryToReadField<inicpp::boolean_ini_t>(content, "Simulation", "show_q_vel", SHOW_Q_VEL);
    SHOW_Q_BIAS = tryToReadField<inicpp::boolean_ini_t>(content, "Simulation", "show_q_bias", SHOW_Q_BIAS);
    SHOW_QC = tryToReadField<inicpp::boolean_ini_t>(content, "Simulation", "show_qc", SHOW_QC);

    CAM_DISTANCE = tryToReadField<inicpp::float_ini_t>(content, "Simulation", "cam_distance", CAM_DISTANCE);
    CAM_ELEVATION = tryToReadField<inicpp::float_ini_t>(content, "Simulation", "cam_elevation", CAM_ELEVATION);
    CAM_AZIMUTH = tryToReadField<inicpp::float_ini_t>(content, "Simulation", "cam_azimuth", CAM_AZIMUTH);

    KP = tryToReadField<inicpp::float_ini_t>(content, "Control", "KP", KP);
    KD = tryToReadField<inicpp::float_ini_t>(content, "Control", "KD", KD);

    dPos(0) = tryToReadField<inicpp::float_ini_t>(content, "Control", "DX", dPos(0));
    dPos(1) = tryToReadField<inicpp::float_ini_t>(content, "Control", "DY", dPos(1));
    dPos(2) = tryToReadField<inicpp::float_ini_t>(content, "Control", "DZ", dPos(2));

    double roll = tryToReadField<inicpp::float_ini_t>(content, "Control", "D_ROLL", 0);
    double pitch = tryToReadField<inicpp::float_ini_t>(content, "Control", "D_PITCH", 0);
    double yaw = tryToReadField<inicpp::float_ini_t>(content, "Control", "D_YAW", 0);

    auto transform = Eigen::AngleAxisd(M_PI * roll / 180, Vec3d::UnitX()) *
                     Eigen::AngleAxisd(M_PI * pitch / 180, Vec3d::UnitY()) *
                     Eigen::AngleAxisd(M_PI * yaw / 180, Vec3d::UnitZ());
    dRot = transform * dRot;
}

MatXd getMassMatrix(pin::Model &m, pin::Data &d, VecXd &q) {
    MatXd massMatrix = pin::crba(m, d, q);
    massMatrix.triangularView<Eigen::StrictlyLower>() = massMatrix.transpose().triangularView<Eigen::StrictlyLower>();
    return massMatrix;
}

VecXd se3error(MatXd rot, MatXd dRot) {
    MatXd errorRot = rot.transpose() * dRot;
    MatXd errorLog = errorRot.log();
    VecXd error(3);
    error << errorLog(2, 1), errorLog(0, 2), errorLog(1, 0);
    return error;
}

VecXd posError(VecXd pos, MatXd rot, VecXd dPos, MatXd dRot) {
    VecXd totalError(6);
    totalError.segment(0, 3) = dPos - pos;
    totalError.segment(3, 3) = se3error(rot, dRot);
    return totalError;
}

MatXd getJacobian(pin::Model &m, pin::Data &d, VecXd &q, pin::JointIndex jIdx) {
    pin::computeJointJacobians(m, d, q);
    MatXd jac(6, m.nv);
    pin::getJointJacobian(m, d, jIdx, pin::LOCAL_WORLD_ALIGNED, jac);
    return jac;
}

MatXd getJacobianTimeVariation(pin::Model &m, pin::Data &d, VecXd &q, VecXd &v, pin::JointIndex jIdx) {
    pin::computeJointJacobiansTimeVariation(m, d, q, v);
    MatXd dJac(6, m.nv);
    pin::getJointJacobianTimeVariation(m, d, jIdx, pin::LOCAL_WORLD_ALIGNED, dJac);
    return dJac;
}

VecXd getConstraintForces(pin::Model &m, pin::Data &d, VecXd &q, VecXd &v, VecXd &f, MatXd &dRot, VecXd &dPos,
                          VecXd &dVel, VecXd &dAcc, pin::JointIndex jIdx) {
    // Get mass matrix
    MatXd massMatrix = getMassMatrix(m, d, q);

    // Get jacobians
    MatXd jac = getJacobian(m, d, q, jIdx);
    MatXd dJac = getJacobianTimeVariation(m, d, q, v, jIdx);

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

void terminate_on_error(const char *msg) {
    printf("%s\n", msg);
    exit(1);
}

int main(int argc, char **argv) {
    // MuJoCo data structures
    char error[1000];
    memset(error, 0, 1000 * sizeof(char));

    // Read configs
    if (argc > 1) {
        readIniConfig(argv[1]);
    } else {
        readIniConfig(DEFAULT_CONFIG_PATH);
    }

    pinocchio::Model p_m;
    pinocchio::Data p_d;

    mjModel *m;
    mjData *d;
    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;

    if (!exists(URDF_PATH)) {
        printf("%s not has been found!\n", URDF_PATH.c_str());
        return 1;
    }
    if (!exists(MUJOCO_XML_PATH)) {
        printf("%s not has been found!\n", MUJOCO_XML_PATH.c_str());
        return 1;
    }

    // init pinocchio
    pinocchio::urdf::buildModel(URDF_PATH, p_m);
    p_d = pinocchio::Data(p_m);

    // init mujoco data
    m = mj_loadXML(MUJOCO_XML_PATH.c_str(), nullptr, error, 1000);
    d = mj_makeData(m);
    if (error[0]) {
        printf("Error occurred: %s\n", error);
        return 1;
    }

    // init GLFW, create window, make OpenGL context current, request v-sync
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(1200, 900, "Demo", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjv_defaultScene(&scn);

    // set camera position
    cam.distance = CAM_DISTANCE;
    cam.elevation = CAM_ELEVATION;
    cam.azimuth = CAM_AZIMUTH;

    printf("nbody = %d\n", m->nbody);
    printf("nq = %d\n", m->nq);
    printf("nv = %d\n", m->nv);
    printf("njt = %d\n", m->njnt);
    printf("cam.elevation = %f\n"
           "cam.azimuth = %f\n"
           "cam.distance = %f\n",
           cam.elevation, cam.azimuth, cam.distance);

    // define storage vectors
    VecXd q(m->nv);
    VecXd qVel(m->nv);
    VecXd qForce(m->nv);
    VecXd qC(m->nv);

    // create scene and context
    mjv_makeScene(m, &scn, 1000);
    mjr_makeContext(m, &con, mjFONTSCALE_100);

    // init mujoco simulation
    mj_checkPos(m, d);
    mj_checkVel(m, d);
    mj_forward(m, d);
    mj_checkAcc(m, d);
    mju_user_warning = terminate_on_error;

    pinocchio::JointIndex pinObjIdx = p_m.getJointId(PIN_OBJ_NAME);
    printf("%s idx = %ld\n", PIN_OBJ_NAME.c_str(), pinObjIdx);

    int mujObjIdx = mj_name2id(m, mjOBJ_BODY, MUJ_OBJ_NAME.c_str());
    printf("%s idx = %d\n", MUJ_OBJ_NAME.c_str(), mujObjIdx);

    VecXd dVel = VecXd::Zero(6);
    VecXd dAcc = VecXd::Zero(6);

    cout << "Desired position: " << dPos.transpose() << "\n";
    cout << "Desired orientation:\n"
         << dRot << "\n";

    // Main loop
    double prevStump = 0;
    while (!glfwWindowShouldClose(window)) {
        mjtNum sim_start = d->time;
        while (d->time - sim_start < 1.0 / (60.0 * TIMES_SLOWER) && !PAUSED) {
            // copy state from mujoco
            memcpy(q.data(), d->qpos, m->nq * sizeof(mjtNum));
            memcpy(qVel.data(), d->qvel, m->nv * sizeof(mjtNum));
            qForce = -pin::rnea(p_m, p_d, q, qVel, VecXd::Zero(p_m.nv));

            // compute constraint forces
            qC = getConstraintForces(p_m, p_d, q, qVel, qForce, dRot, dPos, dVel, dAcc, pinObjIdx);
            memcpy(d->ctrl, qC.data(), m->nv * sizeof(mjtNum));

            mj_step(m, d);
        }

        if (d->time > MAX_DURATION) {
            break;
        }

        // print telemetry
        if (d->time - prevStump > TELEMETRY_PRINT_DELTA) {
            pin::forwardKinematics(p_m, p_d, q);
            VecXd pE = posError(p_d.oMi[pinObjIdx].translation(), p_d.oMi[pinObjIdx].rotation(), dPos, dRot);

            printf("Time: %.3f", d->time);
            if (SHOW_ERROR) {
                printf(", Error value: %.5f", sqrt(pE.dot(pE)));
                printf(", Error: ");
                printVector(pE.data(), 6, false);
            }
            if (SHOW_Q) {
                printf(", q: ");
                printVector(q.data(), m->nq, false);
            }
            if (SHOW_Q_VEL) {
                printf(", qVel: ");
                printVector(qVel.data(), m->nv, false);
            }
            if (SHOW_Q_BIAS) {
                printf(", qForce: ");
                printVector(qForce.data(), m->nv, false);
            }
            if (SHOW_QC) {
                printf(", qC: ");
                printVector(qC.data(), m->nv, false);
            }
            printf("\n");

            prevStump = d->time;
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
