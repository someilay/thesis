#include <chrono>
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

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "GLFW/glfw3.h"
#include <proxsuite/proxqp/dense/dense.hpp>

#define DEFAULT_CONFIG_PATH "config.ini"
#define ASSERT_WITH_MSG(cond, msg)                                                                                     \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::ostringstream str;                                                                                    \
            str << "Assertion failed, " << __FILE__ << ":" << __LINE__ << " - " << msg << "\n";                        \
            std::cerr << str.str();                                                                                    \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)
#define ASSERT(cond)                                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::ostringstream str;                                                                                    \
            str << "Assertion failed, " << __FILE__ << ":" << __LINE__ << "\n";                                        \
            std::cerr << str.str();                                                                                    \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace pin = pinocchio;
namespace prox = proxsuite::proxqp;

using std::cout;
using std::function;
using std::pair;
using std::string;
using std::vector;

using VecXd = Eigen::VectorXd;
using Vec3d = Eigen::Vector3d;
using Vec4d = Eigen::Vector4d;
using Vec6d = Eigen::Matrix<double, 6, 1>;

using MatXd = Eigen::MatrixXd;
using Mat3d = Eigen::Matrix3d;

using int_idx = pin::JointIndex;

struct _config {
    string mjcfModelPath = "yumi/scene.xml";
    string urdfModelPath = "yumi/yumi.urdf";
    string startObjName = "yumi_link_7_l";
    string endObjName = "yumi_link_7_r";

    double timesSlower = 1;
    bool simPaused = false;
    bool showQaddr = false;
    bool showInitialPos = false;
    double maxDuration = INFINITY;
    double telemetryTimeDelta = 0.1;

    double camDistance = 4;
    double camElevation = -30;
    double camAzimuth = 10;

    double constraintKP = 0;
    double constraintKD = 0;
    double controlKP = 0;
    double controlKD = 0;
    double controlWeight = 1;

    vector<double> q0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    bool bodyFrameEnabled = true;

    Mat3d rotDiff = Mat3d::Identity();
    vector<double> p = {0, 1, 0};
} typedef Config;

struct _indices {
    int_idx startIdx;
    int_idx endIdx;
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

void printVector(VecXd &v, bool newLine = true) { printVector(v.data(), (int)v.size(), newLine); }

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
    struct stat buffer {};
    int fd = stat(name.c_str(), &buffer);
    if (fd > -1)
        close(fd);
    return fd == 0;
}

template <typename... Args> std::string string_format(const std::string &format, Args... args) {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

template <typename T>
T tryToReadField(const inicpp::config &content, string collection, const string field, T defaultValue) {
    try {
        return content[collection][field].get<T>();
    } catch (inicpp::not_found_exception &e) {
        cout << collection << ":" << field << " not found, using default value: " << defaultValue << "\n";
    }
    return defaultValue;
}

template <typename T>
vector<T> tryToReadList(
    const inicpp::config &content, string collection, const string field, vector<T> &defaultVector
) {
    try {
        return content[collection][field].get_list<T>();
    } catch (inicpp::not_found_exception &e) {
        cout << collection << ":" << field << " not found";
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

    // simulation configs
    config.mjcfModelPath =
        tryToReadField<inicpp::string_ini_t>(iniFile, "Simulation", "mjcf_model_path", config.mjcfModelPath);
    config.urdfModelPath =
        tryToReadField<inicpp::string_ini_t>(iniFile, "Simulation", "urdf_model_path", config.urdfModelPath);

    config.bodyFrameEnabled =
        tryToReadField<inicpp::boolean_ini_t>(iniFile, "Simulation", "body_frame_enabled", config.bodyFrameEnabled);
    config.timesSlower = tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "times_slower", config.timesSlower);
    config.simPaused = tryToReadField<inicpp::boolean_ini_t>(iniFile, "Simulation", "sim_paused", config.simPaused);
    config.showQaddr = tryToReadField<inicpp::boolean_ini_t>(iniFile, "Simulation", "show_qaddr", config.showQaddr);
    config.showInitialPos =
        tryToReadField<inicpp::boolean_ini_t>(iniFile, "Simulation", "show_initial_pos", config.showInitialPos);
    config.maxDuration = tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "max_duration", config.maxDuration);
    config.telemetryTimeDelta =
        tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "telemetry_time_delta", config.telemetryTimeDelta);

    config.camDistance = tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "cam_distance", config.camDistance);
    config.camElevation =
        tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "cam_elevation", config.camElevation);
    config.camAzimuth = tryToReadField<inicpp::float_ini_t>(iniFile, "Simulation", "cam_azimuth", config.camAzimuth);

    // control configs
    config.startObjName = tryToReadField<inicpp::string_ini_t>(iniFile, "Control", "start_obj", config.startObjName);
    config.endObjName = tryToReadField<inicpp::string_ini_t>(iniFile, "Control", "end_obj", config.endObjName);

    config.controlKP = tryToReadField<inicpp::float_ini_t>(iniFile, "Control", "KP", config.controlKP);
    config.controlKD = tryToReadField<inicpp::float_ini_t>(iniFile, "Control", "KD", config.controlKD);
    config.controlWeight = tryToReadField<inicpp::float_ini_t>(iniFile, "Control", "weight", config.controlWeight);

    config.q0 = tryToReadList<inicpp::float_ini_t>(iniFile, "Control", "q0", config.q0);

    // constraints configs
    double roll = tryToReadField<inicpp::float_ini_t>(iniFile, "Constraint", "d_roll", 0);
    double pitch = tryToReadField<inicpp::float_ini_t>(iniFile, "Constraint", "d_pitch", 0);
    double yaw = tryToReadField<inicpp::float_ini_t>(iniFile, "Constraint", "d_yaw", 0);
    auto transform = Eigen::AngleAxisd(M_PI * roll / 180, Vec3d::UnitX()) *
                     Eigen::AngleAxisd(M_PI * pitch / 180, Vec3d::UnitY()) *
                     Eigen::AngleAxisd(M_PI * yaw / 180, Vec3d::UnitZ());

    config.rotDiff = transform * config.rotDiff;
    config.p = tryToReadList<inicpp::float_ini_t>(iniFile, "Constraint", "p", config.p);

    config.constraintKP = tryToReadField<inicpp::float_ini_t>(iniFile, "Constraint", "KP", config.constraintKP);
    config.constraintKD = tryToReadField<inicpp::float_ini_t>(iniFile, "Constraint", "KD", config.constraintKD);
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

void initMujScene(
    Config &config, mjvCamera &cam, mjvOption &opt, mjvScene &scn, mjrContext &con, GLFWwindow **window, mjModel *m
) {
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

void qAddrAlignedSame(mjModel *muj_m, pin::Model &pin_m, bool showQaddr) {
    auto frames = pin_m.frames;
    auto joints = pin_m.joints;

    for (auto frame : frames) {
        if (frame.type != pin::FrameType::JOINT)
            continue;
        int_idx pin_idx = pin_m.getJointId(frame.name);
        int muj_idx = mj_name2id(muj_m, mjOBJ_JOINT, frame.name.c_str());

        ASSERT((int)pin_idx != pin_m.njoints);
        ASSERT_WITH_MSG(muj_idx != -1, string_format("mujoco xml does not contain \"%s\" joint!", frame.name.c_str()));

        auto joint = joints[pin_idx];
        ASSERT_WITH_MSG(
            joint.idx_q() == muj_m->jnt_qposadr[muj_idx],
            string_format("pin_qaddr != muj_qaddr for \"%s\" joint!", frame.name.c_str())
        );
        ASSERT_WITH_MSG(
            joint.idx_v() == muj_m->jnt_dofadr[muj_idx],
            string_format("pin_vaddr != muj_vaddr for \"%s\" joint!", frame.name.c_str())
        );

        if (showQaddr)
            cout << frame.name << " -> "
                 << "qaddr = " << joint.idx_q() << ", vaddr = " << joint.idx_v() << "\n";
    }
}

Indices getIndices(Config &config, mjModel *muj_m, pin::Model &pin_m) {
    qAddrAlignedSame(muj_m, pin_m, config.showQaddr);
    auto startIdx = pin_m.getBodyId(config.startObjName);
    auto endIdx = pin_m.getBodyId(config.endObjName);
    ASSERT_WITH_MSG(
        (int)startIdx != pin_m.nframes, string_format("urdf does not contain \"%s\" link!", config.startObjName.c_str())
    );
    ASSERT_WITH_MSG(
        (int)endIdx != pin_m.nframes, string_format("urdf does not contain \"%s\" link!", config.endObjName.c_str())
    );

    return {startIdx, endIdx};
}

Vec6d errorInSE3(const pin::SE3 &t, const pin::SE3 &t_des) {
    Vec6d error = Vec6d::Zero();
    error.segment(0, 3) = t_des.translation() - t.translation();
    error.segment(3, 3) = pin::log3(t.rotation().transpose() * t_des.rotation());
    return error;
}

void forwardKin(pin::Model &m, pin::Data &d, VecXd &q, VecXd &dq, bool computeSecondDerivatives = true) {
    pin::forwardKinematics(m, d, q, dq);
    pin::computeJointJacobians(m, d, q);
    if (computeSecondDerivatives) {
        pin::computeJointJacobiansTimeVariation(m, d, q, dq);
    }
    pin::updateFramePlacements(m, d);
}

void forwardKin(pin::Model &m, pin::Data &d, VecXd &q) {
    pin::forwardKinematics(m, d, q);
    pin::computeJointJacobians(m, d, q);
    pin::updateFramePlacements(m, d);
}

MatXd getMassMatrix(pin::Model &m, pin::Data &d, VecXd &q) {
    MatXd massMatrix = pin::crba(m, d, q);
    massMatrix.triangularView<Eigen::StrictlyLower>() = massMatrix.transpose().triangularView<Eigen::StrictlyLower>();
    return massMatrix;
}

MatXd getJacobian(pin::Model &m, pin::Data &d, VecXd &q, int_idx idx, bool recomputeJacs = true) {
    if (recomputeJacs) {
        forwardKin(m, d, q);
    }
    MatXd jac = MatXd::Zero(6, m.nv);
    pin::getFrameJacobian(m, d, idx, pin::LOCAL_WORLD_ALIGNED, jac);
    return jac;
}

MatXd getJacobianTimeVariation(
    pin::Model &m, pin::Data &d, VecXd &q, VecXd &dq, int_idx idx, bool recomputeJacs = true
) {
    if (recomputeJacs) {
        forwardKin(m, d, q, dq);
    }
    MatXd dJac = MatXd::Zero(6, m.nv);
    pin::getFrameJacobianTimeVariation(m, d, idx, pin::LOCAL_WORLD_ALIGNED, dJac);
    return dJac;
}

pin::Motion getClassicVel(
    pin::Model &m, pin::Data &d, VecXd &q, VecXd &dq, int_idx idx, bool recomputeForwardKin = true
) {
    if (recomputeForwardKin) {
        forwardKin(m, d, q, dq, false);
    }
    auto frame = m.frames[idx];
    auto parent_idx = frame.parent;
    auto local_to_world_t = pin::SE3::Identity();

    local_to_world_t.rotation(d.oMf[idx].rotation());
    return local_to_world_t.act(frame.placement.actInv(d.v[parent_idx]));
}

void setState(mjModel *m, mjData *d, VecXd &q, VecXd &dq) {
    memcpy(q.data(), d->qpos, m->nq * sizeof(mjtNum));
    memcpy(dq.data(), d->qvel, m->nv * sizeof(mjtNum));
}

pair<MatXd, VecXd> getAffineDesc(
    MatXd &jacDiff,
    MatXd &dJacDiff,
    pin::Motion &vel,
    pin::Motion &velDes,
    pin::SE3 &t,
    pin::SE3 &tDes,
    VecXd &dq,
    double kD,
    double kP,
    bool onlyRotation = false,
    bool onlyPosition = false
) {
    if (onlyPosition && onlyRotation) {
        std::ostringstream str;
        str << __FILE__ << ":" << __LINE__ << " either onlyRotation or onlyPosition can be set true!\n";
        std::cerr << str.str();
        std::abort();
    }

    // matrix / vector size
    int nc = (onlyPosition || onlyRotation) ? 3 : 6;
    int nv = (int)dq.size();
    // shift in A/b of rotation part
    int rotShift = onlyRotation ? 0 : 3;

    // difference jacobians / time differentiated jacobians
    MatXd jacDiffLin = jacDiff.block(0, 0, 3, nv);
    MatXd jacDiffRot = jacDiff.block(3, 0, 3, nv);
    MatXd dJacDiffLin = dJacDiff.block(0, 0, 3, nv);
    MatXd dJacDiffRot = dJacDiff.block(3, 0, 3, nv);

    // attitude error
    Vec6d errAll = errorInSE3(t, tDes);
    VecXd err = VecXd::Zero(nc);
    if (onlyPosition) {
        err = errAll.segment(0, 3);
    }
    if (onlyRotation) {
        err = errAll.segment(3, 3);
    }
    if (!(onlyPosition || onlyRotation)) {
        err = errAll.segment(0, 6);
    }

    // useful quantities
    Mat3d skewDesAng = pin::skew(velDes.angular());
    MatXd jacDiffRotDes = tDes.rotation().transpose() * jacDiffRot;
    MatXd dJacDiffRotDes = tDes.rotation().transpose() * (dJacDiffRot - skewDesAng * jacDiffRot);

    // A matrix
    MatXd aMat = MatXd::Zero(nc, nv);
    if (!onlyRotation) {
        aMat.block(0, 0, 3, nv) = jacDiffLin;
    }
    if (!onlyPosition) {
        aMat.block(rotShift, 0, 3, nv) = jacDiffRotDes;
    }

    // b vector
    VecXd b = Vec6d::Zero(nc);
    if (!onlyRotation) {
        b.segment(0, 3) += -dJacDiffLin * dq;
    }
    if (!onlyPosition) {
        b.segment(rotShift, 3) += -dJacDiffRotDes * dq;
    }

    // derivative part
    if (!onlyRotation) {
        b.segment(0, 3) += -kD * (velDes.linear() - vel.linear());
    }
    if (!onlyPosition) {
        b.segment(rotShift, 3) += -kD * tDes.rotation().transpose() * (velDes.angular() - vel.angular());
    }

    // proportional part
    b += -kP * err;

    return {aMat, b};
}

pair<MatXd, VecXd> getConstraintsAffineDesc(
    pin::Model &m,
    pin::Data &d,
    VecXd &q,
    VecXd &dq,
    pin::SE3 &tDiff,
    int_idx startIdx,
    int_idx endIdx,
    double kP,
    double kD,
    bool recomputeForwardKin = true
) {
    if (recomputeForwardKin) {
        forwardKin(m, d, q, dq);
    }

    // start jacobians
    MatXd jacStart = getJacobian(m, d, q, startIdx, false);
    MatXd jacStartLin = jacStart.block(0, 0, 3, m.nv);
    MatXd jacStartRot = jacStart.block(3, 0, 3, m.nv);

    // start time differentiated jacobians
    MatXd dJacStart = getJacobianTimeVariation(m, d, q, dq, startIdx, false);
    MatXd dJacStartLin = dJacStart.block(0, 0, 3, m.nv);
    MatXd dJacStartRot = dJacStart.block(3, 0, 3, m.nv);

    // end jacobian / time differentiated jacobian
    MatXd jacEnd = getJacobian(m, d, q, endIdx, false);
    MatXd dJacEnd = getJacobianTimeVariation(m, d, q, dq, endIdx, false);

    // start/end classic velocities
    auto startVel = getClassicVel(m, d, q, dq, startIdx, false);
    auto endVel = getClassicVel(m, d, q, dq, endIdx, false);

    // start/end attitude
    auto startPos = d.oMf[startIdx];
    auto endPos = d.oMf[endIdx];

    // useful quantities
    Vec3d pWorld = startPos.rotation() * tDiff.translation();
    Mat3d skewPWorld = pin::skew(pWorld);
    Mat3d skewStartAng = pin::skew(startVel.angular());

    // object's end jacobian
    MatXd jacObj = MatXd::Zero(6, m.nv);
    jacObj.block(0, 0, 3, m.nv) = jacStartLin - skewPWorld * jacStartRot;
    jacObj.block(3, 0, 3, m.nv) = jacStartRot;

    // object's end time differentiated jacobian
    MatXd dJacObj = MatXd::Zero(6, m.nv);
    dJacObj.block(0, 0, 3, m.nv) = dJacStartLin - skewPWorld * dJacStartRot - skewStartAng * skewPWorld * jacStartRot;
    dJacObj.block(3, 0, 3, m.nv) = dJacStartRot;

    // differences
    MatXd jacDiff = jacEnd - jacObj;
    MatXd dJacDiff = dJacEnd - dJacObj;
    pin::Motion objVel(startVel.linear() + startVel.angular().cross(pWorld), startVel.angular());
    pin::SE3 objPos = startPos * tDiff;

    return getAffineDesc(jacDiff, dJacDiff, objVel, endVel, objPos, endPos, dq, kD, kP);
}

pair<MatXd, VecXd> getControlAffineDesc(
    pin::Model &m,
    pin::Data &d,
    VecXd &q,
    VecXd &dq,
    pin::SE3 &desPos,
    pin::Motion &desVel,
    pin::Motion &desAcc,
    int_idx idx,
    double kP,
    double kD,
    bool recomputeForwardKin = true,
    bool onlyRotation = false,
    bool onlyPosition = false
) {
    if (recomputeForwardKin) {
        forwardKin(m, d, q, dq);
    }

    // jacobian
    MatXd jac = -getJacobian(m, d, q, idx, false);
    // time differentiated jacobian
    MatXd dJac = -getJacobianTimeVariation(m, d, q, dq, idx, false);

    // velocity & position
    auto vel = getClassicVel(m, d, q, dq, idx, false);
    auto pos = d.oMf[idx];

    auto [aMat, bVec] = getAffineDesc(jac, dJac, vel, desVel, pos, desPos, dq, kD, kP, onlyRotation, onlyPosition);
    // subtract desired acceleration from b
    if (!onlyRotation) {
        bVec.segment(0, 3) += -desAcc.linear();
    }
    if (!onlyPosition) {
        int rotShift = onlyRotation ? 0 : 3;
        bVec.segment(rotShift, 3) += desPos.rotation().transpose() * desAcc.angular();
    }

    return {aMat, bVec};
}

VecXd getControlForces(
    pin::Model &m,
    pin::Data &d,
    VecXd &q,
    VecXd &dq,
    VecXd &biasForces,
    pin::SE3 &tDiff,
    pin::SE3 &desPos,
    pin::Motion &desVel,
    pin::Motion &desAcc,
    Indices &indices,
    Config &config,
    prox::dense::QP<double> &qp,
    bool &firstCall,
    double eps_abs = 1e-4,
    int *elapsedTime = nullptr,
    bool onlyRotation = false,
    bool onlyPosition = false
) {
    // end measuring computation time
    auto begin = std::chrono::steady_clock::now();

    // perform forward kinematics
    forwardKin(m, d, q, dq);

    // compute mass matrix
    MatXd massMat = getMassMatrix(m, d, q);

    auto [aObjMat, bObjVec] = getConstraintsAffineDesc(
        m, d, q, dq, tDiff, indices.startIdx, indices.endIdx, config.constraintKP, config.constraintKD, false
    );

    auto [aConMat, bConVec] = getControlAffineDesc(
        m,
        d,
        q,
        dq,
        desPos,
        desVel,
        desAcc,
        indices.startIdx,
        config.controlKP,
        config.controlKD,
        false,
        onlyRotation,
        onlyPosition
    );

    qp.settings.eps_abs = eps_abs;
    qp.settings.initial_guess =
        firstCall ? prox::InitialGuessStatus::NO_INITIAL_GUESS : prox::InitialGuessStatus::WARM_START;
    qp.settings.verbose = false;
    firstCall = false;

    qp.init(
        massMat + config.controlWeight * aConMat.transpose() * aConMat,
        -biasForces - config.controlWeight * aConMat.transpose() * bConVec,
        aObjMat,
        bObjVec,
        proxsuite::nullopt,
        proxsuite::nullopt,
        proxsuite::nullopt
    );
    qp.solve();

    if (qp.results.info.status != prox::QPSolverOutput::PROXQP_SOLVED) {
        std::ostringstream str;
        str << __FILE__ << ":" << __LINE__ << " QP cannot be solved!\n";
        std::cerr << str.str();
        std::abort();
    }
    // getting optimal acceleration
    VecXd res = pin::rnea(m, d, q, dq, qp.results.x);

    // end measuring computation time
    auto end = std::chrono::steady_clock::now();
    if (elapsedTime) {
        *elapsedTime = (int)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    }

    return res;
}

pin::SE3 getDesPos(double t, double w, double r) {
    Vec3d tr = {0.536763 - 0.1, 0.185684, 0.412661 + 0.1 + r * sin(w * t)};
    return pin::SE3(Mat3d::Identity(), tr);
}

pin::Motion getDesVel(double t, double w, double r) {
    Vec3d lin = {0, 0, r * w * cos(w * t)};
    return pin::Motion(lin, Vec3d::Zero());
}

pin::Motion getDesAcc(double t, double w, double r) {
    Vec3d lin = {0, 0, -r * w * w * sin(w * t)};
    return pin::Motion(lin, Vec3d::Zero());
}

void mainLoop(
    Config &config,
    mjvCamera &cam,
    mjvOption &opt,
    mjvScene &scn,
    mjrContext &con,
    GLFWwindow *window,
    Indices &indices,
    mjModel *muj_m,
    mjData *muj_d,
    pin::Model &pin_m,
    pin::Data &pin_d
) {
    // model state
    VecXd q(pin_m.nq);
    VecXd dq(pin_m.nv);
    // bias forces
    VecXd q_bias(pin_m.nv);
    // constraints force
    VecXd qc(pin_m.nv);
    // define QP
    bool firstCallOfQp = true;
    prox::dense::QP<double> qp(pin_m.nv, 6, 0);

    // SE3 difference between start and end of object
    Vec3d p = Vec3d::Zero();
    for (size_t i = 0; i < 3; i++) {
        p[i] = config.p[i];
    }
    auto tDiff = pin::SE3(config.rotDiff, p);

    // Elapsed time
    int meanElapsedTime = 0;
    int elapsedTime = 0;
    int cntElapsed = 0;

    // show inital link position if needed
    if (config.showInitialPos) {
        setState(muj_m, muj_d, q, dq);
        pin::framesForwardKinematics(pin_m, pin_d, q);
        auto startPos = pin_d.oMf[indices.startIdx];
        auto endPos = pin_d.oMf[indices.endIdx];
        cout << config.startObjName << " position:\n";
        cout << startPos << "\n";
        cout << config.endObjName << " position:\n";
        cout << endPos << "\n";
    }

    // if pause show message
    if (config.simPaused) {
        cout << "PAUSED!!!\n";
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

        meanElapsedTime = 0;
        cntElapsed = 0;
        while (muj_d->time - sim_start < 1.0 / (60.0 * config.timesSlower) && !config.simPaused) {
            setState(muj_m, muj_d, q, dq);
            q_bias = -pin::rnea(pin_m, pin_d, q, dq, VecXd::Zero(pin_m.nv));

            auto desPos = getDesPos(muj_d->time, 1, 0.12);
            auto desVel = getDesVel(muj_d->time, 1, 0.12);
            auto desAcc = getDesAcc(muj_d->time, 1, 0.12);

            qc = getControlForces(
                pin_m,
                pin_d,
                q,
                dq,
                q_bias,
                tDiff,
                desPos,
                desVel,
                desAcc,
                indices,
                config,
                qp,
                firstCallOfQp,
                1e-4,
                &elapsedTime
            );
            memcpy(muj_d->ctrl, qc.data(), pin_m.nv * sizeof(mjtNum));

            meanElapsedTime += elapsedTime;
            cntElapsed += 1;

            mj_step(muj_m, muj_d);
        }
        meanElapsedTime = (int)((double)meanElapsedTime / cntElapsed);

        if (muj_d->time > config.maxDuration) {
            break;
        }

        // print telemetry
        if (muj_d->time - prevStump > config.telemetryTimeDelta) {
            auto desPos = getDesPos(muj_d->time, 1, 0.12);

            printf("Time: %.3f", muj_d->time);
            printf(", QP elapsed time: %d mcs", meanElapsedTime);

            // get frame positions
            pin::SE3 tStart = pin_d.oMf[indices.startIdx];
            pin::SE3 tEnd = pin_d.oMf[indices.endIdx];
            // get errors
            Vec6d eStart = errorInSE3(tStart * tDiff, tEnd);
            Vec6d eEnd = errorInSE3(tStart, desPos);

            printf(", eConstrain: %.5f", eStart.dot(eStart));
            printf(", eControl: %.5f", eEnd.dot(eEnd));

            printf(", left arm pos: ");
            printVector(tStart.translation().data(), tStart.translation().size());

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
    ASSERT_WITH_MSG(muj_m->nq == pin_m.nq, "Pinocchio and MuJoCo has different DOF!");
    ASSERT_WITH_MSG(muj_m->nv == pin_m.nv, "Pinocchio and MuJoCo has different DOF!");
    if (config.q0.size())
        ASSERT_WITH_MSG((int)config.q0.size() == pin_m.nq, "q0 zero should have the right dimensionality");

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
