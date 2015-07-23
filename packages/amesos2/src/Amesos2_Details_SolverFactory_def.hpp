// @HEADER
//
// ***********************************************************************
//
//           Amesos2: Templated Direct Sparse Solver Package
//                  Copyright 2011 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

/// \file   Amesos2_Details_SolverFactory_def.hpp
/// \author Mark Hoemmen
/// \brief  Definition of Amesos2::Details::SolverFactory.

#ifndef AMESOS2_DETAILS_SOLVERFACTORY_DEF_HPP
#define AMESOS2_DETAILS_SOLVERFACTORY_DEF_HPP

#include "Amesos2_config.h"
#include "Amesos2_Factory.hpp"
#include "Amesos2_Solver.hpp"
#include "Trilinos_Details_Solver.hpp"
#include "Trilinos_Details_SolverFactory.hpp"
#include <type_traits>

#ifdef HAVE_AMESOS2_EPETRA
#  include "Epetra_CrsMatrix.h"
#endif // HAVE_AMESOS2_EPETRA

// mfh 23 Jul 2015: Tpetra is currently a required dependency of Amesos2.
#ifndef HAVE_AMESOS2_TPETRA
#  define HAVE_AMESOS2_TPETRA
#endif // HAVE_AMESOS2_TPETRA

#ifdef HAVE_AMESOS2_TPETRA
#  include "Tpetra_CrsMatrix.hpp"
#endif // HAVE_AMESOS2_TPETRA

namespace Amesos2 {
namespace Details {

// Amesos2 unfortunately only does ETI for Tpetra::CrsMatrix, even
// though it could very well take Tpetra::RowMatrix.
template<class OP>
struct GetMatrixType {
  typedef int type; // flag (see below)

#ifdef HAVE_AMESOS2_EPETRA
  static_assert(! std::is_same<OP, Epetra_MultiVector>::value,
                "Amesos2::Details::GetMatrixType: OP = Epetra_MultiVector.  "
                "This probably means that you mixed up MV and OP.");
#endif // HAVE_AMESOS2_EPETRA

#ifdef HAVE_AMESOS2_TPETRA
  static_assert(! std::is_same<OP, Tpetra::MultiVector<typename OP::scalar_type,
                typename OP::local_ordinal_type, typename OP::global_ordinal_type,
                typename OP::node_type> >::value,
                "Amesos2::Details::GetMatrixType: OP = Tpetra::MultiVector.  "
                "This probably means that you mixed up MV and OP.");
#endif // HAVE_AMESOS2_TPETRA
};

#ifdef HAVE_AMESOS2_EPETRA
template<>
struct GetMatrixType<Epetra_Operator> {
  typedef Epetra_CrsMatrix type;
};
#endif // HAVE_AMESOS2_EPETRA

#ifdef HAVE_AMESOS2_TPETRA
template<class S, class LO, class GO, class NT>
struct GetMatrixType<Tpetra::Operator<S, LO, GO, NT> > {
  typedef Tpetra::CrsMatrix<S, LO, GO, NT> type;
};
#endif // HAVE_AMESOS2_TPETRA

template<class MV, class OP>
class LinearSolver : public Trilinos::Details::Solver<MV, OP> {
#ifdef HAVE_AMESOS2_EPETRA
  static_assert(! std::is_same<OP, Epetra_MultiVector>::value,
                "Amesos2::Details::LinearSolver: OP = Epetra_MultiVector.  "
                "This probably means that you mixed up MV and OP.");
  static_assert(! std::is_same<MV, Epetra_Operator>::value,
                "Amesos2::Details::LinearSolver: MV = Epetra_Operator.  "
                "This probably means that you mixed up MV and OP.");
#endif // HAVE_AMESOS2_EPETRA

public:
  /// \brief Type of the CrsMatrix specialization corresponding to OP.
  ///
  /// Amesos2 currently requires a CrsMatrix as its template
  /// parameter, not an Operator or a RowMatrix.  However, the
  /// SolverFactory system is templated on Operator.  This type traits
  /// class converts from OP to the corresponding CrsMatrix type.
  typedef typename GetMatrixType<OP>::type crs_matrix_type;
  static_assert(! std::is_same<crs_matrix_type, int>::value,
                "Amesos2::Details::LinearSolver: The given OP type is not "
                "supported.");

  //! Type of the underlying Amesos2 solver.
  typedef Amesos2::Solver<crs_matrix_type, MV> solver_type;

  /// \brief Constructor.
  ///
  /// \param solverName [in] Name of the solver to create.  Must be a
  ///   name accepted by Amesos2::create.
  ///
  /// \note To developers: The constructor takes a name rather than a
  ///   solver, because Amesos2 does not currently let us create a
  ///   solver with a null matrix.  To get around this, we create a
  ///   new solver whenever the matrix changes.  The unfortunate side
  ///   effect is that, since Amesos2 has no way of determining
  ///   whether a solver supports a given set of template parameters
  ///   without attempting to create the solver, we can't actually
  ///   test in the constructor whether the solver exists.
  LinearSolver (const std::string& solverName) : solverName_ (solverName) {}

  //! Destructor (virtual for memory safety).
  virtual ~LinearSolver () {}

  /// \brief Set the Solver's matrix.
  ///
  /// \param A [in] Pointer to the matrix A in the linear system(s)
  ///   AX=B to solve.
  void setMatrix (const Teuchos::RCP<const OP>& A) {
    using Teuchos::null;
    using Teuchos::RCP;
    typedef crs_matrix_type MAT;

    if (A.is_null ()) {
      solver_ = Teuchos::null;
    }
    else {
      RCP<const MAT> A_mat = Teuchos::rcp_dynamic_cast<const MAT> (A);
      TEUCHOS_TEST_FOR_EXCEPTION
        (A_mat.is_null (), std::invalid_argument,
         "Input matrix A must be a CrsMatrix");
      if (solver_.is_null ()) {
        // Amesos2 solvers must be created with a nonnull matrix.
        // Thus, we don't actually create the solver until setMatrix
        // is called for the first time with a nonnull matrix.
        // Furthermore, Amesos2 solvers do not accept a null matrix
        // (to setA), so if setMatrix is called with a null matrix, we
        // free the solver.
        RCP<solver_type> solver =
          Amesos2::create<MAT, MV> (solverName_, A_mat, null, null);
        // Use same parameters as before, if user set parameters.
        if (! params_.is_null ()) {
          solver->setParameters (params_);
        }
        solver_ = solver;
      } else if (A_ != A) {
        solver_->setA (A_mat);
      }
    }

    // We also have to keep a pointer to A, so that getMatrix() works.
    A_ = A;
  }

  //! Get a pointer to this Solver's matrix.
  Teuchos::RCP<const OP> getMatrix () const {
    return A_;
  }

  //! Solve the linear system(s) AX=B.
  void solve (MV& X, const MV& B) {
    TEUCHOS_TEST_FOR_EXCEPTION
      (solver_.is_null (), std::runtime_error, "Amesos2::Details::LinearSolver::"
       "solve: The solver does not exist yet.  You must call setMatrix() "
       "with a nonnull matrix before you may call this method.");
    TEUCHOS_TEST_FOR_EXCEPTION
      (A_.is_null (), std::runtime_error, "Amesos2::Details::LinearSolver::"
       "solve: The matrix has not been set yet.  You must call setMatrix() "
       "with a nonnull matrix before you may call this method.");
    solver_->solve (&X, &B);
  }

  //! Set this solver's parameters.
  void setParameters (const Teuchos::RCP<Teuchos::ParameterList>& params) {
    if (! solver_.is_null ()) {
      solver_->setParameters (params);
    }
    // Remember them, so that if the solver gets recreated, we'll have
    // the original parameters.
    params_ = params;
  }

  /// \brief Set up any part of the solve that depends on the
  ///   structure of the input matrix, but not its numerical values.
  void symbolic () {
    TEUCHOS_TEST_FOR_EXCEPTION
      (solver_.is_null (), std::runtime_error, "Amesos2::Details::LinearSolver::"
       "symbolic: The solver does not exist yet.  You must call setMatrix() "
       "with a nonnull matrix before you may call this method.");
    TEUCHOS_TEST_FOR_EXCEPTION
      (A_.is_null (), std::runtime_error, "Amesos2::Details::LinearSolver::"
       "symbolic: The matrix has not been set yet.  You must call setMatrix() "
       "with a nonnull matrix before you may call this method.");
    solver_->symbolicFactorization ();
  }

  /// \brief Set up any part of the solve that depends on both the
  ///   structure and the numerical values of the input matrix.
  void numeric () {
    TEUCHOS_TEST_FOR_EXCEPTION
      (solver_.is_null (), std::runtime_error, "Amesos2::Details::LinearSolver::"
       "numeric: The solver does not exist yet.  You must call setMatrix() "
       "with a nonnull matrix before you may call this method.");
    TEUCHOS_TEST_FOR_EXCEPTION
      (A_.is_null (), std::runtime_error, "Amesos2::Details::LinearSolver::"
       "numeric: The matrix has not been set yet.  You must call setMatrix() "
       "with a nonnull matrix before you may call this method.");
    solver_->numericFactorization ();
  }

private:
  std::string solverName_;
  Teuchos::RCP<solver_type> solver_;
  Teuchos::RCP<const OP> A_;
  Teuchos::RCP<Teuchos::ParameterList> params_;
};

template<class MV, class OP>
Teuchos::RCP<Trilinos::Details::Solver<MV, OP> >
SolverFactory<MV, OP>::getSolver (const std::string& solverName)
{
  return Teuchos::rcp (new Amesos2::Details::LinearSolver<MV, OP> (solverName));
}

} // namespace Details
} // namespace Amesos2

#endif // AMESOS2_DETAILS_SOLVERFACTORY_DEF_HPP
