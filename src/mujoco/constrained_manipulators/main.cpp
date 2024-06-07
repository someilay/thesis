#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "inicpp.h"
#include "mujoco/mujoco.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "GLFW/glfw3.h"

#define DEFAULT_CONFIG_PATH "config.ini"

namespace pin = pinocchio;

using std::cout;
using std::pair;
using std::string;
using std::vector;

using VecXd = Eigen::VectorXd;
using Vec3d = Eigen::Vector3d;
using Vec4d = Eigen::Vector4d;
using Vec6d = Eigen::Matrix<double, 6, 1>;

using MatXd = Eigen::MatrixXd;
using Mat3d = Eigen::Matrix3d;

struct _config {
    string mjcfModelPath = "mjcf_manipulators/scene.xml";
    string urdfModelPath = "urdf_manipulators.urdf";

    string mjcfStartObj = "1_gripper";
    string mjcfEndObj = "2_gripper";
    string urdfStartObj = "1_gripper-body_5-joint";
    string urdfEndObj = "2_gripper-body_5-joint";

    double timesSlower = 1;
    bool simPaused = false;
    double maxDuration = INFINITY;
    double telemetryTimeDelta = 0.1;

    double camDistance = 4;
    double camElevation = -30;
    double camAzimuth = 10;

    double kP = 0;
    double kD = 0;

    vector<double> q0;
    bool bodyFrameEnabled = true;

    Mat3d rotDiff = Mat3d::Identity();
    vector<double> p = {0, 1, 0}; 
} typedef Config;

struct _indices {
    int mjcfStartIdx;
    int mjcfEndIdx;
    pin::JointIndex urdfStartIdx;
    pin::JointIndex urdfEndIdx;
} typedef Indices;

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

void printVector(VecXd &v, bool newLine = true) {
    printVector(v.data(), (int)v.size(), newLine);
}

void printMatrix(MatXd &mat) {
    int rows = (int)mat.rows();
    int cols = (int)mat.cols();
    for (int j = 0; j < rows; ++j) {
        for (int k = 0; k < cols; ++k) {
            if (mat(j, k) >= 0)
                printf(" %.4f ", mat(j, k));
            else
                printf("%.4f ", mat(j, k));
        }
        printf("\n");
    }
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

template <typename T>
vector<T> tryToReadList(const inicpp::config &content, string collection, const string field, vector<T> &defaultVector) {
    try {
        return content[collection][field].get_list<T>();
    } catch (inicpp::not_found_exception &e) {
        cout << e.what();
        if (defaultVector.size()) {
            cout << ", using default value: ";
            for (size_t i = 0; i < defaultVector.size() - 1; i++) {
                cout << defaultVector[i] << ", ";
            }
            cout << defaultVector[defaultVector.size() - 1];
        }
        cout << "\n";
    }
    return defaultVector;
}

void initConfig(int argc, char **argv, Config &config) {
    string configPath = DEFAULT_CONFIG_PATH;
    if (argc > 1) {
        configPath = argv[1];
    }
    if (!exists(configPath)) {
        printf("ini-file not found, using default settings\n");
        return;
    }

    inicpp::config iniFile = inicpp::parser::load_file(configPath);

    config.mjcfModelPath = tryToReadField<inicpp::string_ini_t>(iniFile, "Simulation", "mjcf_model_path", config.mjcfModelPath);
    config.urdfModelPath = tryToReadField<inicpp::string_ini_t>(iniFile, "Simulation", "urdf_model_path", config.urdfModelPath);

    config.mjcfStartObj = tryToReadField<inicpp::string_ini_t>(iniFile, "Control", "mjcf_start_obj", config.mjcfStartObj);
    config.mjcfEndObj = tryToReadField<inicpp::string_ini_t>(iniFile, "Control", "mjcf_end_obj", config.mjcfEndObj);
    config.urdfStartObj = tryToReadField<inicpp::string_ini_t>(iniFile, "Control", "urdf_start_obj", config.urdfStartObj);
    config.urdfEndObj = tryToReadField<inicpp::string_ini_t>(iniFile, "Control", "urdf_end_obj", config.urdfEndObj);

    config.kP = tryToReadField<inicpp::float_ini_t>(iniFile, "Control", "KP", config.kP);
    config.kD = tryToReadField<inicpp::float_ini_t>(iniFile, "Control", "KD", config.kD);

    config.bodyFrameEnabled = tryToReadField<inicpp::boolean_ini_t>(iniFile, "Simulation", "body_frame_enabled", config.bodyFrameEnabled);
    config.timesSlower = tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "times_slower", config.timesSlower);
    config.simPaused = tryToReadField<inicpp::boolean_ini_t>(iniFile, "Simulation", "sim_paused", config.simPaused);
    config.maxDuration = tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "max_duration", config.maxDuration);
    config.telemetryTimeDelta = tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "telemetry_time_delta", config.telemetryTimeDelta);

    config.camDistance = tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "cam_distance", config.camDistance);
    config.camElevation = tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "cam_elevation", config.camElevation);
    config.camAzimuth = tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "cam_azimuth", config.camAzimuth);

    config.q0 = tryToReadList<inicpp::float_ini_t>(iniFile, "Control", "q0", config.q0);

    double roll = tryToReadField<inicpp::float_ini_t>(iniFile, "Control", "d_roll", 0);
    double pitch = tryToReadField<inicpp::float_ini_t>(iniFile, "Control", "d_pitch", 0);
    double yaw = tryToReadField<inicpp::float_ini_t>(iniFile, "Control", "d_yaw", 0);
    auto transform = Eigen::AngleAxisd(M_PI * roll / 180, Vec3d::UnitX()) *
                     Eigen::AngleAxisd(M_PI * pitch / 180, Vec3d::UnitY()) *
                     Eigen::AngleAxisd(M_PI * yaw / 180, Vec3d::UnitZ());

    config.rotDiff = transform * config.rotDiff;
    config.p = tryToReadList<inicpp::float_ini_t>(iniFile, "Control", "p", config.p);
}

void loadMjcfModel(mjModel **m, mjData **d, string mjcfModelPath) {
    char muj_error[1000];
    memset(muj_error, 0, 1000 * sizeof(char));

    if (!exists(mjcfModelPath)) {
        printf("%s not has been found!\n", mjcfModelPath.c_str());
        exit(1);
    }

    *m = mj_loadXML(mjcfModelPath.c_str(), nullptr, muj_error, 1000);
    *d = mj_makeData(*m);
    if (muj_error[0]) {
        printf("Error occurred: %s\n", muj_error);
        exit(1);
    }
}

void loadUrdfModel(pin::Model &m, pin::Data &d, string urdfModelPath) {
    if (!exists(urdfModelPath)) {
        printf("%s not has been found!\n", urdfModelPath.c_str());
        exit(1);
    }

    pin::urdf::buildModel(urdfModelPath, m);
    d = pin::Data(m);
}

void initMujScene(Config &config,
                  mjvCamera &cam,
                  mjvOption &opt,
                  mjvScene &scn,
                  mjrContext &con,
                  GLFWwindow **window,
                  mjModel *m) {
    // init GLFW, create window, make OpenGL context current, request v-sync
    glfwInit();
    *window = glfwCreateWindow(1200, 900, "Constrained manipulators", nullptr, nullptr);
    glfwMakeContextCurrent(*window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjv_defaultScene(&scn);

    // set camera position
    cam.distance = config.camDistance;
    cam.elevation = config.camElevation;
    cam.azimuth = config.camAzimuth;

    // create scene and context
    mjv_makeScene(m, &scn, 1000);
    mjr_makeContext(m, &con, mjFONTSCALE_100);

    if (config.bodyFrameEnabled)
        opt.frame = mjFRAME_BODY;
}

void terminate_on_error(const char *msg) {
    printf("%s\n", msg);
    exit(1);
}

pair<int, int> getMjcfIndices(mjModel *m, string mjcfStartObj, string mjcfEndObj) {
    int startIdx = mj_name2id(m, mjOBJ_BODY, mjcfStartObj.c_str());
    int endIdx = mj_name2id(m, mjOBJ_BODY, mjcfEndObj.c_str());

    if (startIdx == -1) {
        printf("%s not found in mjcf description!\n", mjcfStartObj.c_str());
        exit(1);
    }
    if (endIdx == -1) {
        printf("%s not found in mjcf description!\n", mjcfEndObj.c_str());
        exit(1);
    }

    return std::make_pair(startIdx, endIdx);
}

pair<pin::JointIndex, pin::JointIndex> getUrdfIndices(pin::Model &m, string urdfStartObj, string urdfEndObj) {
    auto startIdx = m.getJointId(urdfStartObj);
    auto endIdx = m.getJointId(urdfEndObj);

    if (startIdx == (pin::JointIndex) m.njoints) {
        printf("%s not found in urdf description!\n", urdfStartObj.c_str());
        exit(1);
    }
    if (endIdx == (pin::JointIndex) m.njoints) {
        printf("%s not found in urdf description!\n", urdfEndObj.c_str());
        exit(1);
    }

    return std::make_pair(startIdx, endIdx);
}

Indices getIndices(Config &config, mjModel *muj_m, pin::Model &pin_m) {
    auto [mjcfStartIdx, mjcfEndIdx] = getMjcfIndices(muj_m, config.mjcfStartObj, config.mjcfEndObj);
    auto [urdfStartIdx, urdfEndIdx] = getUrdfIndices(pin_m, config.urdfStartObj, config.urdfEndObj);
    return {
        mjcfStartIdx,
        mjcfEndIdx,
        urdfStartIdx,
        urdfEndIdx};
}

Vec6d errorInSE3(const pin::SE3 &t, const pin::SE3 &t_des) {
    Vec6d error = Vec6d::Zero();
    error.segment(0, 3) = t_des.translation() - t.translation();
    error.segment(3, 3) = pin::log3(t.rotation().transpose() * t_des.rotation());
    return error;
}

Vec6d computeSE3Error(pin::Model &m,
                      pin::Data &d,
                      Indices &indices,
                      VecXd &q,
                      Mat3d &rotDiff,
                      Vec3d &p,
                      bool forwardKinematicsWasComputed = false) {
    if (!forwardKinematicsWasComputed)
        pin::forwardKinematics(m, d, q);

    pin::SE3 t_start = d.oMi[indices.urdfStartIdx];
    pin::SE3 t_end = d.oMi[indices.urdfEndIdx];
    pin::SE3 t_diff;

    t_diff.setIdentity();
    t_diff.rotation(rotDiff);
    t_diff.translation(p);

    return errorInSE3(t_start * t_diff, t_end);
}

MatXd getMassMatrix(pin::Model &m, pin::Data &d, VecXd &q) {
    MatXd massMatrix = pin::crba(m, d, q);
    massMatrix.triangularView<Eigen::StrictlyLower>() = massMatrix.transpose().triangularView<Eigen::StrictlyLower>();
    return massMatrix;
}

MatXd getJacobian(pin::Model &m, pin::Data &d, VecXd &q, pin::JointIndex jIdx, bool recomputeJacs = true) {
    if (recomputeJacs)
        pin::computeJointJacobians(m, d, q);
    MatXd jac = MatXd::Zero(6, m.nv);
    pin::getJointJacobian(m, d, jIdx, pin::LOCAL_WORLD_ALIGNED, jac);
    return jac;
}

MatXd getJacobianTimeVariation(pin::Model &m, pin::Data &d, VecXd &q, VecXd &dq, pin::JointIndex jIdx, bool recomputeJacs = true) {
    if (recomputeJacs)
        pin::computeJointJacobiansTimeVariation(m, d, q, dq);
    MatXd dJac = MatXd::Zero(6, m.nv);
    pin::getJointJacobianTimeVariation(m, d, jIdx, pin::LOCAL_WORLD_ALIGNED, dJac);
    return dJac;
}

void setState(mjModel *m, mjData *d, VecXd &q, VecXd &dq) {
    memcpy(q.data(), d->qpos, m->nq * sizeof(mjtNum));
    memcpy(dq.data(), d->qvel, m->nv * sizeof(mjtNum));
}

VecXd getConstraintForces(pin::Model &m,
                          pin::Data &d,
                          VecXd &q,
                          VecXd &dq,
                          VecXd &f,
                          Mat3d &rotDiff,
                          Vec3d &p,
                          Indices &indices,
                          Config &config) {
    // Get mass matrix
    MatXd massMatrix = getMassMatrix(m, d, q);

    // Get jacobians
    MatXd jacStart = getJacobian(m, d, q, indices.urdfStartIdx);
    MatXd jacEnd = getJacobian(m, d, q, indices.urdfEndIdx, false);
    MatXd dJacStart = getJacobianTimeVariation(m, d, q, dq, indices.urdfStartIdx);
    MatXd dJacEnd = getJacobianTimeVariation(m, d, q, dq, indices.urdfEndIdx, false);

    // Get useful blocks
    MatXd jacVelDiff = (jacEnd - jacStart).block(0, 0, 3, m.nv);
    MatXd jacRotDiff = (jacEnd - jacStart).block(3, 0, 3, m.nv);
    MatXd dJacVelDiff = (dJacEnd - dJacStart).block(0, 0, 3, m.nv);
    MatXd dJacRotDiff = (dJacEnd - dJacStart).block(3, 0, 3, m.nv);
    MatXd jacRotStart = jacStart.block(3, 0, 3, m.nv); 
    MatXd dJacRotStart = dJacStart.block(3, 0, 3, m.nv); 

    // Compute current orientation
    Mat3d rotStart = d.oMi[indices.urdfStartIdx].rotation();
    Mat3d rotEnd = d.oMi[indices.urdfEndIdx].rotation();

    // Compute spatial velocities
    Vec6d vSpatialStart = jacStart * dq;
    Vec6d vSpatialEnd = jacEnd * dq;

    // Get spatial angular velocities' skews
    Vec3d omegaStart = vSpatialStart.segment(3, 3);
    Vec3d omegaEnd = vSpatialEnd.segment(3, 3);
    Mat3d skOmegaStart = pin::skew(omegaStart);
    Mat3d skOmegaEnd = pin::skew(omegaEnd);

    // Compute angular velocities in the end frame
    Vec3d omegaStartEnd = rotEnd.transpose() * omegaStart;
    Vec3d omegaEndEnd = rotEnd.transpose() * omegaEnd;

    // Other useful blocks
    Vec3d pSpatial = rotStart * p;
    Mat3d skPS = pin::skew(pSpatial);
    MatXd dP = rotEnd.transpose() * (dJacRotDiff - skOmegaEnd * jacRotDiff);

    // Compute linear velocities of linking body and end frame
    Vec3d velStart = vSpatialStart.segment(0, 3);
    Vec3d velBody = omegaStart.cross(pSpatial) + velStart;
    Vec3d velEnd = vSpatialEnd.segment(0, 3);

    // Compute M^-1, M^(1 / 2) and M^(-1 / 2)
    MatXd invM = massMatrix.inverse();
    MatXd sqrtM = massMatrix.sqrt();
    MatXd invSqrtM = sqrtM.inverse();

    // Compute A
    MatXd a = MatXd::Zero(6, m.nv);
    a.block(0, 0, 3, m.nv) = jacVelDiff + skPS * jacRotStart;
    a.block(3, 0, 3, m.nv) = rotEnd.transpose() * jacRotDiff; // P

    // Compute AM^-1 and [AM^(-1 / 2)]^+
    MatXd aM = a * invM;
    MatXd pAiSqrtM = (a * invSqrtM).completeOrthogonalDecomposition().pseudoInverse();

    // Compute error
    Vec6d pE = computeSE3Error(m, d, indices, q, rotDiff, p, true);
    Vec6d dpE = Vec6d::Zero();
    dpE.segment(0, 3) = velEnd - velBody;
    dpE.segment(3, 3) = omegaEndEnd - omegaStartEnd;

    // Compute b
    Vec3d bLinDrift = -dJacVelDiff * dq - skPS * dJacRotStart * dq + skOmegaStart * skOmegaStart * pSpatial;
    Vec3d bAngDrift = -dP * dq;
    Vec6d b = -config.kP * pE - config.kD * dpE;
    b.segment(0, 3) += bLinDrift;
    b.segment(3, 3) += bAngDrift;

    // Compute Q_c
    VecXd fC = sqrtM * pAiSqrtM * (b - aM * f);

    return fC;
}

void mainLoop(Config &config,
              mjvCamera &cam,
              mjvOption &opt,
              mjvScene &scn,
              mjrContext &con,
              GLFWwindow *window,
              Indices &indices,
              mjModel *muj_m,
              mjData *muj_d,
              pin::Model &pin_m,
              pin::Data &pin_d) {
    // model state
    VecXd q(pin_m.nq);
    VecXd dq(pin_m.nv);
    // bias forces
    VecXd q_bias(pin_m.nv);
    // constraints force
    VecXd qc(pin_m.nv);

    Vec3d p = Vec3d::Zero();
    for (size_t i = 0; i < 3; i++) {
        p[i] = config.p[i];
    }

    // init mujoco simulation
    mj_checkPos(muj_m, muj_d);
    mj_checkVel(muj_m, muj_d);
    mj_forward(muj_m, muj_d);
    mj_checkAcc(muj_m, muj_d);
    mju_user_warning = terminate_on_error;

    // Main loop
    double prevStump = 0;
    while (!glfwWindowShouldClose(window)) {
        mjtNum sim_start = muj_d->time;
        while (muj_d->time - sim_start < 1.0 / (60.0 * config.timesSlower) && !config.simPaused) {
            setState(muj_m, muj_d, q, dq);
            q_bias = -pin::rnea(pin_m, pin_d, q, dq, VecXd::Zero(pin_m.nv));

            qc = getConstraintForces(pin_m, pin_d, q, dq, q_bias, config.rotDiff, p, indices, config);
            memcpy(muj_d->ctrl, qc.data(), pin_m.nv * sizeof(mjtNum));

            mj_step(muj_m, muj_d);
        }

        if (muj_d->time > config.maxDuration) {
            break;
        }

        // print telemetry
        if (muj_d->time - prevStump > config.telemetryTimeDelta) {
            VecXd pE = computeSE3Error(pin_m, pin_d, indices, q, config.rotDiff, p, false);

            printf("Time: %.3f", muj_d->time);
            printf(", pE: ");
            printVector(pE.data(), pE.size(), false);
            printf(", |pE|^2: %.5f", pE.dot(pE));
            printf("\n");

            prevStump = muj_d->time;
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(muj_m, muj_d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
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
}

int main(int argc, char **argv) {
    Config config;
    initConfig(argc, argv, config);

    // MuJoCo data structures
    mjModel *muj_m = nullptr;
    mjData *muj_d = nullptr;
    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;

    // Pinocchio data structures
    pin::Model pin_m;
    pin::Data pin_d;

    // GLFW data structures
    GLFWwindow *window = nullptr;

    // init MuJoCo data
    loadMjcfModel(&muj_m, &muj_d, config.mjcfModelPath);

    // init Pinocchio data
    loadUrdfModel(pin_m, pin_d, config.urdfModelPath);

    // check correspondence
    assert(("Pinocchio and MuJoCo has different DOF!", muj_m->nq == pin_m.nq));
    assert(("Pinocchio and MuJoCo has different DOF!", muj_m->nv == pin_m.nv));
    if (config.q0.size())
        assert(("q0 zero should have the right dimensionality", (int)config.q0.size() == pin_m.nq));

    // set initial MuJoCo state
    if (config.q0.size())
        mju_copy(muj_d->qpos, &config.q0[0], pin_m.nv);

    // get indices
    Indices indices = getIndices(config, muj_m, pin_m);

    // init MuJoCo scene
    initMujScene(config, cam, opt, scn, con, &window, muj_m);

    // start a loop
    mainLoop(config, cam, opt, scn, con, window, indices, muj_m, muj_d, pin_m, pin_d);

    // free memory
    free(muj_m);
    free(muj_d);
    return 0;
}
