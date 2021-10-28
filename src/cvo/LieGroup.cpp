#include "cvo/LieGroup.h"
#include <unsupported/Eigen/MatrixFunctions>
///#include <opencv2/core/eigen.hpp>

using namespace std;

namespace cvo {

  const float TOLERANCE = 1e-6;

  template <typename T, int RC_MAJOR>
  Eigen::Matrix<T, 3, 3, RC_MAJOR> skew(const Eigen::Matrix<T, 3, 1>& v) {
    // Convert vector to skew-symmetric matrix
    Eigen::Matrix<T, 3, 3, RC_MAJOR> M = Eigen::Matrix<T, 3, 3, RC_MAJOR>::Zero();
    M << 0, -v[2], v[1],
      v[2], 0, -v[0], 
      -v[1], v[0], 0;
    return M;
  }
  template Eigen::Matrix<double, 3, 3, Eigen::RowMajor> skew<double, Eigen::RowMajor>(const Eigen::Matrix<double, 3, 1> & v);
  template Eigen::Matrix<float, 3, 3, Eigen::ColMajor> skew<float, Eigen::ColMajor>(const Eigen::Matrix<float, 3, 1, Eigen::ColMajor> & v);

  Eigen::Vector3f unskew(const Eigen::Matrix3f& M){
    Eigen::Vector3f v = Eigen::Vector3f::Zero();
    v << M(2,1), M(0,2), M(1,0);
    return v;
  }

  Eigen::Matrix4f hat2(const Eigen::VectorXf& x){
    Eigen::Matrix4f X = Eigen::Matrix4f::Zero();
    X.block<3,3>(0,0) = skew<float, Eigen::ColMajor>(x.head(3));
    X.block<3,1>(0,3) = x.tail(3);
    return X;
  }

  Eigen::VectorXf wedge(const Eigen::Matrix4f& X){
    Eigen::VectorXf x(6);
    x.head(3) = unskew(X.block<3,3>(0,0));
    x.tail(3) = X.block<3,1>(0,3);
    return x;
  }


  template <typename T, int RC_MAJOR>
  Eigen::Matrix<T, 3, 3, RC_MAJOR> LeftJacobian_SO3(const Eigen::Matrix<T, 3, 1>& w){
    Eigen::Matrix<T, 3, 3, RC_MAJOR> A = skew<T, RC_MAJOR>(w);
    T theta = w.norm();
    if (theta < TOLERANCE) {
      return Eigen::Matrix<T, 3,3, RC_MAJOR>::Identity();
    } 
    // eye(3) + ((1-cos(theta))/theta^2)*A + ((theta-sin(theta))/theta^3)*A^2;
    Eigen::Matrix<T,3,3, RC_MAJOR> R =  Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity() + ((1-cos(theta))/(theta*theta))*A \
      + ((theta-sin(theta))/(theta*theta*theta))*A*A;
    return R;
  }
  template 
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> LeftJacobian_SO3<double, Eigen::RowMajor>(const Eigen::Matrix<double, 3, 1>& w);
  template
  Eigen::Matrix<float, 3, 3, Eigen::ColMajor> LeftJacobian_SO3<float, Eigen::ColMajor>(const Eigen::Matrix<float, 3, 1>& w);


  Eigen::MatrixXf LeftJacobianInverse_SO3(const Eigen::Vector3f& w){
    Eigen::Matrix3f A = skew<float, Eigen::ColMajor>(w);
    float theta = w.norm();
    if (theta < TOLERANCE) {
      return Eigen::Matrix3f::Identity();
    } 
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity()-0.5*A+((1.0/(theta*theta))-(1+cos(theta))/(2.0*theta*sin(theta)))*A*A;
    return R;
  }

  Eigen::MatrixXf LeftJacobian_SE3(Eigen::VectorXf& v){
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(6,6);
    Eigen::VectorXf Phi = v.head(3);
    float phi = Phi.norm();
    Eigen::VectorXf Rho = v.tail(3);
    Eigen::Matrix3f Phi_skew = skew<float, Eigen::ColMajor>(Phi);
    Eigen::Matrix3f Rho_skew = skew<float, Eigen::ColMajor>(Rho);
    Eigen::Matrix3f J = LeftJacobian_SO3<float, Eigen::ColMajor>(Phi);
    Eigen::Matrix3f Q = Eigen::Matrix3f::Zero();

    if(phi < TOLERANCE)
      Q = 0.5*Rho_skew;
    else
    {   
      float phi2 = phi*phi;
      float phi3 = phi*phi*phi;
      float phi4 = phi*phi*phi*phi;
      float phi5 = phi*phi*phi*phi*phi;
      Q = 0.5*Rho_skew\
        + (phi-sin(phi))/phi3 * (Phi_skew*Rho_skew + Rho_skew*Phi_skew + Phi_skew*Rho_skew*Phi_skew)\
        - (1-0.5*phi2-cos(phi))/phi4 * (Phi_skew*Phi_skew*Rho_skew + Rho_skew*Phi_skew*Phi_skew \
                                        - 3*Phi_skew*Rho_skew*Phi_skew) - 0.5*((1-0.5*phi2-cos(phi))/phi4 \
                                                                               - 3*(phi-sin(phi)-(phi3)/6)/phi5) \
        * (Phi_skew*Rho_skew*Phi_skew*Phi_skew + Phi_skew*Phi_skew*Rho_skew*Phi_skew);
    }
    output.block<3,3>(0,0) = J;
    output.block<3,3>(0,3) = Q;
    output.block<3,3>(3,3) = J;
    
    return output;
  }

  Eigen::MatrixXf RightJacobian_SE3(Eigen::VectorXf& v){
    if(v.norm() < TOLERANCE)
      return Eigen::MatrixXf::Identity(6,6);
    else
      return Adjoint_SEK3(hat2(-v).exp()) * LeftJacobian_SE3(v);
  }


  Eigen::MatrixXf RightJacobianInverse_SE3(Eigen::VectorXf& v){
    Eigen::MatrixXf Jr = RightJacobian_SE3(v);
    if(v.norm() < TOLERANCE)
      return Eigen::MatrixXf::Identity(6,6);
    else
      return Jr.inverse()*Eigen::MatrixXf::Identity(6,6);
  }


  Eigen::Vector3f Log_SO3(const Eigen::Matrix3f& M){
    float theta = acos((M.trace()-1)/2);
    if(theta<TOLERANCE){
      return Eigen::Vector3f::Zero();
    }
    return unskew(theta*(M-M.transpose())/(2*sin(theta)));
  }

  Eigen::VectorXf Log_SE3(const Eigen::MatrixXf& X){
    Eigen::Vector3f w = Log_SO3(X.block<3,3>(0,0));
    Eigen::Vector3f v = LeftJacobianInverse_SO3(w)*X.block<3,1>(0,3);
    Eigen::VectorXf s(6);
    s << w , v;

    return s;

  }


  Eigen::MatrixXf Exp_SE3(const Eigen::VectorXf& v){
    Eigen::Vector3f w = v.head(3);
    Eigen::Vector3f u = v.tail(3);
    Eigen::MatrixXf X = Eigen::Matrix4f::Identity();
    X.block<3,3>(0,0) = Exp_SO3<float, Eigen::ColMajor>(w);
    X.block<3,1>(0,3) = LeftJacobian_SO3<float, Eigen::ColMajor>(w)*u;
    return X;
  }

  template <typename T, int RC_MAJOR>
  Eigen::Matrix<T, 3, 3, RC_MAJOR> Exp_SO3(const Eigen::Matrix<T, 3, 1>& w) {
    // Computes the vectorized exponential map for SO(3)
    Eigen::Matrix<T, 3, 3, RC_MAJOR> A = skew<T, RC_MAJOR>(w);
    auto theta = w.norm();
    if (theta < TOLERANCE) {
      return Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity();
    } 
    Eigen::Matrix<T, 3, 3, RC_MAJOR> R =  Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity() + (sin(theta)/theta)*A + ((1-cos(theta))/(theta*theta))*A*A;
    return R;
  }
  template 
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Exp_SO3<double, Eigen::RowMajor>(const Eigen::Matrix<double, 3, 1>& w);
  template
  Eigen::Matrix<float, 3, 3, Eigen::ColMajor> Exp_SO3<float, Eigen::ColMajor>(const Eigen::Matrix<float, 3, 1>& w);


  

  template <typename T, int RC_MAJOR>
  Eigen::Matrix<T, 3, 4, RC_MAJOR> Exp_SE3(const Eigen::Matrix<T, 6, 1>& v, bool is_wu){
    
    Eigen::Matrix<T, 3, 1> w;
    w = v.head(3);
    Eigen::Matrix<T, 3, 1> u;
    u = v.tail(3);
    if (is_wu) {
      w = v.head(3);
      u = v.tail(3);
    } else {
      w = v.tail(3);
      u = v.head(3);
    }
    
    typename Eigen::Matrix<T, 3, 4, RC_MAJOR> X;
    Eigen::Matrix<T, 3, 3, RC_MAJOR> X_R;
    X_R= Exp_SO3<T,  RC_MAJOR>(w);
    X.template block<3,3>(0,0) = X_R;
    
    Eigen::Matrix<T, 3, 1> X_T;
    X_T = LeftJacobian_SO3<T, RC_MAJOR>(w)*u;
    X.template block<3,1>(0,3) = X_T;
    return X;
  }
  template 
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> Exp_SE3(const Eigen::Matrix<double, 6, 1>& v,
                                                       bool is_wu
                                                       );
  template
  Eigen::Matrix<float, 3, 4, Eigen::ColMajor> Exp_SE3(const Eigen::Matrix<float, 6, 1>& v,
                                                       bool is_wu
                                                       );


  Eigen::Matrix3f Exp_SO3(const Eigen::Vector3f& w) {
    // Computes the vectorized exponential map for SO(3)
    Eigen::Matrix3f A = skew<float, Eigen::ColMajor>(w);
    float theta = w.norm();
    if (theta < TOLERANCE) {
      return Eigen::Matrix3f::Identity();
    } 
    Eigen::Matrix3f R =  Eigen::Matrix3f::Identity() + (sin(theta)/theta)*A + ((1-cos(theta))/(theta*theta))*A*A;
    return R;
  }

  Eigen::MatrixXf Exp_SEK3(const Eigen::VectorXf& v, float dt) {
    // Computes the vectorized exponential map for SE_K(3)
    int K = (v.size()-3)/3;
    Eigen::MatrixXf X = Eigen::MatrixXf::Identity(3+K,3+K);
    Eigen::Matrix3f R;
    Eigen::Matrix3f Jl;
    Eigen::Vector3f w = v.head(3);
    float theta = w.norm();
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    if (theta < TOLERANCE) {
      R = I;
      Jl = I;
    } else {
      Eigen::Matrix3f A = skew<float, Eigen::ColMajor>(w);
      float theta2 = theta*theta;
      float stheta = sin(dt*theta);
      float ctheta = cos(dt*theta);
      float oneMinusCosTheta2 = (1-ctheta)/(theta2);
      Eigen::Matrix3f A2 = A*A;
      R =  I + (stheta/theta)*A + oneMinusCosTheta2*A2;
      Jl = dt*I + oneMinusCosTheta2*A + ((dt*theta-stheta)/(theta2*theta))*A2;
    }
    X.block<3,3>(0,0) = R;
    for (int i=0; i<K; ++i) {
      X.block<3,1>(0,3+i) = Jl * v.segment<3>(3+3*i);
    }
    return X;
  }


  __attribute__((force_align_arg_pointer))
  Eigen::Matrix<float, 3, 4> Exp_SEK3(const Eigen::Matrix<float, 6,1>& v, float dt) {
    // Computes the vectorized exponential map for SE_K(3)
    Eigen::Matrix3f R;
    Eigen::Matrix3f Jl;
    Eigen::Vector3f w = v.head(3);
    float theta = w.norm();
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    if (theta < TOLERANCE) {
      R = I;
      Jl = I;
    } else {
      Eigen::Matrix3f A = skew<float, Eigen::ColMajor>(w);
      float theta2 = theta*theta;
      float stheta = sin(dt*theta);
      float ctheta = cos(dt*theta);
      float oneMinusCosTheta2 = (1-ctheta)/(theta2);
      Eigen::Matrix3f A2 = A*A;
      R =  I + (stheta/theta)*A + oneMinusCosTheta2*A2;
      Jl = dt*I + oneMinusCosTheta2*A + ((dt*theta-stheta)/(theta2*theta))*A2;
    }
    Eigen::Matrix<float, 3, 4> X;
    X(0,0) = 1.0; X(1,1) = 1.0;  X(2,2) = 1.0;
    X.block<3,3>(0,0) = R;
    for (int i=0; i<1; ++i) {
      Eigen::Vector3f v3=  v.segment<3>(3+3*i);
      X.block<3,1>(0,3+i) = Jl * v3;
    }
    return X;

  }

  Eigen::MatrixXf Adjoint_SEK3(const Eigen::MatrixXf& X) {
    // Compute Adjoint(X) for X in SE_K(3)
    int K = X.cols()-3;
    Eigen::MatrixXf Adj = Eigen::MatrixXf::Zero(3+3*K, 3+3*K);
    Eigen::Matrix3f R = X.block<3,3>(0,0);
    Adj.block<3,3>(0,0) = R;
    for (int i=0; i<K; ++i) {
      Adj.block<3,3>(3+3*i,3+3*i) = R;
      Adj.block<3,3>(3+3*i,0) = skew<float, Eigen::ColMajor>(X.block<3,1>(0,3+i))*R;
    }
    return Adj;
  }

  
  Eigen::Vector3cf poly_solver_order3(const Eigen::Vector4f& coef){
    // extract order
    int order = 3;
    Eigen::Vector3cf roots;
    
    // create M = diag(ones(n-1,1),-1)
    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    M.bottomLeftCorner(2,2).setIdentity(); //= Eigen::Matrix2f::Identity();
    
    // M(1,:) = -p(2:n+1)./p(1)
    M.row(0) = -(coef/coef(0)).segment(1,order).transpose();

    // eigen(M) and get the answer
    roots = M.eigenvalues();

    return roots;
  }

  
  Eigen::Vector3cd poly_solver_order3(const Eigen::Matrix<double, 4, 1, Eigen::DontAlign>& coef){
    // extract order
    int order = 3;
    Eigen::Vector3cd roots;
    
    // create M = diag(ones(n-1,1),-1)
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    M.bottomLeftCorner(2,2).setIdentity(); //= Eigen::Matrix2f::Identity();
    
    // M(1,:) = -p(2:n+1)./p(1)
    M.row(0) = -(coef/coef(0)).segment(1,order).transpose();

    // eigen(M) and get the answer
    roots = M.eigenvalues();

    return roots;
  }



  Eigen::VectorXcf poly_solver(const Eigen::VectorXf& coef){
    // extract order
    int order = coef.size()-1;
    Eigen::VectorXcf roots;
    
    // create M = diag(ones(n-1,1),-1)
    Eigen::MatrixXf M = Eigen::MatrixXf::Zero(order,order);
    M.bottomLeftCorner(order-1,order-1) = Eigen::MatrixXf::Identity(order-1,order-1);
    
    // M(1,:) = -p(2:n+1)./p(1)
    M.row(0) = -(coef/coef(0)).segment(1,order).transpose();

    // eigen(M) and get the answer
    roots = M.eigenvalues();

    return roots;
  }


  Eigen::VectorXcd poly_solver(const Eigen::VectorXd& coef){
    // extract order
    int order = coef.size()-1;
    Eigen::VectorXcd roots;
    
    // create M = diag(ones(n-1,1),-1)
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(order,order);
    M.bottomLeftCorner(order-1,order-1) = Eigen::MatrixXd::Identity(order-1,order-1);
    
    // M(1,:) = -p(2:n+1)./p(1)
    M.row(0) = -(coef/coef(0)).segment(1,order).transpose();

    // eigen(M) and get the answer
    roots = M.eigenvalues();

    return roots;
  }


  __attribute__((force_align_arg_pointer))
  double dist_se3(const Eigen::Matrix3f& R, const Eigen::Vector3f& T)  {
    // create transformation matrix
    //Eigen::Matrix4d temp_transform ;
    Eigen::Matrix4f temp_transform  = Eigen::Matrix4f::Identity();
    temp_transform.block<3,3>(0,0)=R;
    temp_transform.block<3,1>(0,3)=T;
    (temp_transform)(3,3) = 1.0;
    float d = temp_transform.log().norm();
    return d;
  }

  __attribute__((force_align_arg_pointer))
  double dist_se3(const Eigen::Matrix3d& R, const Eigen::Vector3d& T)  {
    // create transformation matrix
    //printf("Size of matrix4f is %d\n", sizeof(Eigen::Matrix4f));
    Eigen::Matrix4d temp_transform ;
    //Eigen::Matrix4f temp_transform;// = Eigen::Matrix4f::Identity();
    temp_transform.block<3,3>(0,0)=R;
    temp_transform.block<3,1>(0,3)=T;
    (temp_transform)(3,3) = 1.0;
    // distance = frobenius_norm(logm(trans))
    auto lie_alg_v = temp_transform.log();
    double d = (temp_transform.log().norm());
    //printf("Transform is %f, %f, %f, %f, %f, %f, %f, %f...\n,", temp_transform(0,0), temp_transform(0,1), temp_transform(0,2), temp_transform(0,3),
    //		  temp_transform(1,0), temp_transform(1,1), temp_transform(1,2), temp_transform(1,3));
    //printf("R.log() is  %.4lf,%.4lf,%.4lf\n", lie_alg_v(1,1), lie_alg_v(1,2), lie_alg_v(2,2) );
    return d;
  }
}
