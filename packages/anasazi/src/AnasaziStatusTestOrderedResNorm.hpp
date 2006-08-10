
// @HEADER
// ***********************************************************************
//
//                 Belos: Block Linear Solvers Package
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
// @HEADER
//

#ifndef ANASAZI_STATUS_TEST_ORDEREDRESNORM_HPP
#define ANASAZI_STATUS_TEST_ORDEREDRESNORM_HPP

/*!
  \file AnasaziStatusTestOrderedResNorm.hpp
  \brief A status test for testing the norm of the eigenvectors residuals along with a 
         set of auxiliary eigenvalues.
*/


#include "AnasaziStatusTest.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_LAPACK.hpp"

  /*! 
    \class Anasazi::StatusTestOrderedResNorm 
    
    \brief A status test for testing the norm of the eigenvectors residuals
    along with a set of auxiliary eigenvalues. 
    
    The test evaluates to ::Passed when then the most significant of the
    eigenvalues all have a residual below a certain threshhold. The purpose of
    the test is to not only test convergence for some number of eigenvalues,
    but to test convergence for the correct ones.
    
    In addition to specifying the tolerance, the user may specify:
    <ul>
      <li> the norm to be used: 2-norm or OrthoManager::norm() or getRitzRes2Norms()
      <li> the scale: absolute or relative to magnitude of Ritz value 
      <li> the quorum: the number of vectors required for the test to 
           evaluate as ::Passed.
    </ul>

    Finally, the user must specify the Anasazi::SortManager used for deciding
    significance. 
  */

namespace Anasazi {


template <class ScalarType, class MV, class OP>
class StatusTestOrderedResNorm : public StatusTest<ScalarType,MV,OP> {

 private:
  typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
  typedef Teuchos::ScalarTraits<MagnitudeType> MT;

 public:

  //! @name Enums
  //@{

  /*! \enum ResType 
      \brief Enumerated type used to specify which residual norm used by this status test.
   */
  enum ResType {
    RES_ORTH,
    RES_2NORM,
    RITZRES_2NORM
  };

  //@}

  //! @name Constructors/destructors
  //@{ 

  //! Constructor
  StatusTestOrderedResNorm(Teuchos::RefCountPtr<SortManager<ScalarType,MV,OP> > sorter, MagnitudeType tol, int quorum = -1, ResType whichNorm = RES_ORTH, bool scaled = true);

  //! Destructor
  virtual ~StatusTestOrderedResNorm() {};
  //@}

  //! @name Status methods
  //@{ 
  /*! Check status as defined by test.
    \return TestStatus indicating whether the test passed or failed.
  */
  TestStatus checkStatus( Eigensolver<ScalarType,MV,OP>* solver );

  //! Return the result of the most recent checkStatus call.
  TestStatus getStatus() const { return _state; }
  //@}

  //! @name Reset methods
  //@{ 
  //! Informs the status test that it should reset its internal configuration to the uninitialized state.
  /*! This is necessary for the case when the status test is being reused by another solver or for another
    eigenvalue problem. The status test may have information that pertains to a particular problem or solver 
    state. The internal information will be reset back to the uninitialized state. The user specified information 
    that the convergence test uses will remain.
  */
  void reset() { 
    _state = Undefined;
  }

  //! Clears the results of the last status test.
  /*! This should be distinguished from the reset() method, as it only clears the cached result from the last 
   * status test, so that a call to getStatus() will return ::Undefined. This is necessary for the SEQOR and SEQAND
   * tests in the StatusTestCombo class, which may short circuit and not evaluate all of the StatusTests contained
   * in them.
  */
  void clearStatus() {
    _state = Undefined;
  }

  /*! \brief Set the auxiliary eigenvalues.
   *
   *  This routine sets only the real part of the auxiliary eigenvalues; the imaginary part is set to zero. This routine also resets the state to ::Undefined.
   */
  void setAuxVals(const std::vector<MagnitudeType> &vals) {
    _rvals = vals;
    _ivals.resize(_rvals.size(),MT::zero());
    _state = Undefined;
  }

  /*! \brief Set the auxiliary eigenvalues.
   *
   *  This routine sets both the real and imaginary parts of the auxiliary eigenvalues. This routine also resets the state to ::Undefined.
   */
  void setAuxVals(const std::vector<MagnitudeType> &rvals, const std::vector<MagnitudeType> &ivals) {
    _rvals = rvals;
    _ivals = ivals;
    _state = Undefined;
  }

  //@}

  //! @name Accessor methods
  //@{ 

  /*! \brief Set tolerance.
   *  This also resets the test status to ::Undefined.
   */
  void setTolerance(MagnitudeType tol) {
    TEST_FOR_EXCEPTION(tol <= MT::zero(), StatusTestError, "StatusTestOrderedResNorm: test tolerance must be strictly positive.");
    _state = Undefined;
    _tol = tol;
  }

  //! Get tolerance.
  MagnitudeType getTolerance() {return _tol;}

  /*! \brief Set the residual norm to be used by the status test.
   *
   *  This also resets the test status to ::Undefined.
   */
  void setWhichNorm(ResType whichNorm) {
    _state = Undefined;
    _whichNorm = whichNorm;
  }

  //! Return the residual norm used by the status test.
  ResType getWhichNorm() {return _whichNorm;}

  /*! \brief Instruct test to scale norms by eigenvalue estimates (relative scale).
   *  This also resets the test status to ::Undefined.
   */
  void setScale(bool relscale) {
    _state = Undefined;
    _scaled = relscale;
  }

  //! Returns true if the test scales the norms by the eigenvalue estimates (relative scale).
  bool getScale() {return _scaled;}

  //! Get the indices for the vectors that passed the test.
  std::vector<int> whichVecs() {
    return _ind;
  }

  //! Get the number of vectors that passed the test.
  int howMany() {
    return _ind.size();
  }

  //@}

  //! @name Print methods
  //@{ 
  
  //! Output formatted description of stopping test to output stream.
  ostream& print(ostream& os, int indent = 0) const;
 
  //@}
  private:
    TestStatus _state;
    MagnitudeType _tol;
    std::vector<int> _ind;
    int _quorum;
    bool _scaled;
    ResType _whichNorm;
    std::vector<MagnitudeType> _rvals, _ivals;
    Teuchos::RefCountPtr<SortManager<ScalarType,MV,OP> > _sorter;
};


template <class ScalarType, class MV, class OP>
StatusTestOrderedResNorm<ScalarType,MV,OP>::StatusTestOrderedResNorm(Teuchos::RefCountPtr<SortManager<ScalarType,MV,OP> > sorter, MagnitudeType tol, int quorum, ResType whichNorm, bool scaled)
  : _state(Undefined), _quorum(quorum), _scaled(scaled), _whichNorm(whichNorm), _sorter(sorter) 
{
  TEST_FOR_EXCEPTION(_sorter == Teuchos::null, StatusTestError, "StatusTestOrderedResNorm::constructor() was passed null pointer for SortManager.");
  setTolerance(tol); 
}

template <class ScalarType, class MV, class OP>
TestStatus StatusTestOrderedResNorm<ScalarType,MV,OP>::checkStatus( Eigensolver<ScalarType,MV,OP>* solver ) {


  // get the eigenvector/ritz residuals norms (using the appropriate norm)
  // get the eigenvalues/ritzvalues as well
  std::vector<MagnitudeType> res; 
  std::vector<MagnitudeType> vals = solver->getRitzValues();
  std::vector<int> ind = solver->getRitzIndex();
  switch (_whichNorm) {
    case RES_2NORM:
      res = solver->getRes2Norms();
      vals.resize(res.size());
      break;
    case RES_ORTH:
      res = solver->getResNorms();
      vals.resize(res.size());
      break;
    case RITZRES_2NORM:
      res = solver->getRitzRes2Norms();
      break;
  }

  int numaux = _rvals.size();
  int bs = res.size();
  int num = bs + numaux;

  if (num == 0) {
    _ind.resize(0);
    return Failed;
  }

  // extract the real and imaginary parts from the 
  std::vector<MagnitudeType> allrvals(bs), allivals(bs);
  for (int i=0; i<bs; i++) {
    if (ind[i] == 0) {
      allrvals[i] = vals[i];
      allivals[i] = MT::zero();
    }
    else if (ind[i] == +1) {
      allrvals[i] = vals[i];
      allivals[i] = vals[i+1];
    }
    else if (ind[i] == -1) {
      allrvals[i] =  vals[i-1];
      allivals[i] = -vals[i];
    }
    else {
      TEST_FOR_EXCEPTION(true,std::logic_error,"Anasazi::StatusTestOrderedResNorm::checkStatus(): invalid index returned from getRitzIndex().");
    }
  }

  // put the auxiliary values in the vectors as well
  allrvals.insert(allrvals.end(),_rvals.begin(),_rvals.end());
  allivals.insert(allivals.end(),_ivals.begin(),_ivals.end());

  // if appropriate, scale the norms by the magnitude of the eigenvalue estimate
  Teuchos::LAPACK<int,MagnitudeType> lapack;
  if (_scaled) {
    for (unsigned int i=0; i<res.size(); i++) {
      MagnitudeType tmp = lapack.LAPY2(allrvals[i],allivals[i]);
      if ( tmp != MT::zero() ) {
        res[i] /= tmp;
      }
    }
  }
  // add -1 residuals for the auxiliary values (because -1 < _tol)
  res.insert(res.end(),numaux,-MT::one());

  // we don't actually need the sorted eigenvalues; just the permutation vector
  std::vector<int> perm(num,-1);
  _sorter->sort(solver,num,allrvals,allivals,&perm);

  // apply the sorting to the residuals and original indices
  std::vector<MagnitudeType> oldres = res;
  for (int i=0; i<num; i++) {
    res[i] = oldres[perm[i]];
  }

  // indices: [0,bs) are from solver, [bs,bs+numaux) are from auxiliary values
  _ind.resize(num);

  // test the norms: we want res [0,quorum) to be <= tol
  int have = 0;
  int need = (_quorum == -1) ? num : _quorum;
  int tocheck = need > num ? num : need;
  for (int i=0; i<tocheck; i++) {
    TEST_FOR_EXCEPTION( MT::isnaninf(res[i]), StatusTestError, "StatusTestOrderedResNorm::checkStatus(): residual norm is nan or inf" );
    if (res[i] < _tol) {
      _ind[have] = perm[i];
      have++;
    }
  }
  _ind.resize(have);
  _state = (have >= need) ? Passed : Failed;
  return _state;
}


template <class ScalarType, class MV, class OP>
ostream& StatusTestOrderedResNorm<ScalarType,MV,OP>::print(ostream& os, int indent) const {
  string ind(indent,' ');
  os << ind << "- StatusTestOrderedResNorm: ";
  switch (_state) {
  case Passed:
    os << "Passed" << endl;
    break;
  case Failed:
    os << "Failed" << endl;
    break;
  case Undefined:
    os << "Undefined" << endl;
    break;
  }
  os << ind << "(Tolerance,WhichNorm,Scaled,Quorum): " 
            << "(" << _tol;
  switch (_whichNorm) {
  case RES_ORTH:
    os << ",RES_ORTH";
    break;
  case RES_2NORM:
    os << ",RES_2NORM";
    break;
  case RITZRES_2NORM:
    os << ",RITZRES_2NORM";
    break;
  }
  os        << "," << (_scaled   ? "true" : "false")
            << "," << _quorum 
            << ")" << endl;
  os << ind << "Auxiliary values: ";
  if (_rvals.size() > 0) {
    for (unsigned int i=0; i<_rvals.size(); i++) {
      os << "(" << _rvals[i] << ", " << _ivals[i] << ")  ";
    }
    os << endl;
  }
  else {
    os << "[empty]" << endl;
  }

  if (_state != Undefined) {
    os << ind << "Which vectors: ";
    if (_ind.size() > 0) {
      for (unsigned int i=0; i<_ind.size(); i++) os << _ind[i] << " ";
      os << endl;
    }
    else {
      os << "[empty]" << endl;
    }
  }
  return os;
}


} // end of Anasazi namespace

#endif /* ANASAZI_STATUS_TEST_ORDEREDRESNORM_HPP */
