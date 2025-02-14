// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Tpetra_TestingUtilities.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_Details_KokkosCounter.hpp"
#include "Kokkos_ArithTraits.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_DefaultSerialComm.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_TypeNameTraits.hpp"
#include <iterator>

// FINISH: add test for MultiVector with a node containing zero local entries
// FINISH: add tests for local MultiVectors


// Macro that marks a function as "possibly unused," in order to
// suppress build warnings.
#if ! defined(TRILINOS_UNUSED_FUNCTION)
#  if defined(__GNUC__) || (defined(__INTEL_COMPILER) && !defined(_MSC_VER))
#    define TRILINOS_UNUSED_FUNCTION __attribute__((__unused__))
#  elif defined(__clang__)
#    if __has_attribute(unused)
#      define TRILINOS_UNUSED_FUNCTION __attribute__((__unused__))
#    else
#      define TRILINOS_UNUSED_FUNCTION
#    endif // Clang has 'unused' attribute
#  elif defined(__IBMCPP__)
// IBM's C++ compiler for Blue Gene/Q (V12.1) implements 'used' but not 'unused'.
//
// http://pic.dhe.ibm.com/infocenter/compbg/v121v141/index.jsp
#    define TRILINOS_UNUSED_FUNCTION
#  else // some other compiler
#    define TRILINOS_UNUSED_FUNCTION
#  endif
#endif // ! defined(TRILINOS_UNUSED_FUNCTION)


namespace Teuchos {
  template <>
  ScalarTraits<int>::magnitudeType
  relErr( const int &s1, const int &s2 ) {
    typedef ScalarTraits<int> ST;
    return ST::magnitude(s1-s2);
  }

  template <>
  ScalarTraits<char>::magnitudeType
  relErr( const char &s1, const char &s2 ) {
    typedef ScalarTraits<char> ST;
    return ST::magnitude(s1-s2);
  }
}

namespace {

  using Tpetra::TestingUtilities::getDefaultComm;

  using std::endl;
  using std::copy;
  using std::string;

  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::arrayView;
  using Teuchos::as;
  using Teuchos::Comm;
  using Teuchos::null;
  using Teuchos::Range1D;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::REDUCE_MIN;
  using Teuchos::reduceAll;
  using Teuchos::OrdinalTraits;
  using Teuchos::outArg;
  using Teuchos::ScalarTraits;
  using Teuchos::SerialDenseMatrix;
  using Teuchos::Tuple;
  using Teuchos::tuple;
  using Teuchos::NO_TRANS;
  using Teuchos::TRANS;
  using Teuchos::CONJ_TRANS;
  using Teuchos::VERB_DEFAULT;
  using Teuchos::VERB_NONE;
  using Teuchos::VERB_LOW;
  using Teuchos::VERB_MEDIUM;
  using Teuchos::VERB_HIGH;
  using Teuchos::VERB_EXTREME;

  using Tpetra::Map;
  using Tpetra::MultiVector;
  using Tpetra::global_size_t;
  using Tpetra::GloballyDistributed;

  using Tpetra::createContigMapWithNode;
  using Tpetra::createLocalMapWithNode;

  double errorTolSlack = 1.0e+2;

  TEUCHOS_STATIC_SETUP()
  {
    Teuchos::CommandLineProcessor &clp = Teuchos::UnitTestRepository::getCLP();
    clp.addOutputSetupOptions(true);
    clp.setOption(
        "test-mpi", "test-serial", &Tpetra::TestingUtilities::testMpi,
        "Test MPI (if available) or force test of serial.  In a serial build,"
        " this option is ignored and a serial comm is always used." );
    clp.setOption(
        "error-tol-slack", &errorTolSlack,
        "Slack off of machine epsilon used to check test results" );
  }

  // no ScalarTraits<>::eps() for integer types
  template <class Scalar>
  typename Teuchos::ScalarTraits<Scalar>::magnitudeType testingTol() { return Teuchos::ScalarTraits<Scalar>::eps(); }
  template <>
  TRILINOS_UNUSED_FUNCTION int testingTol<int>() { return 0; }
  template <>
  TRILINOS_UNUSED_FUNCTION long testingTol<long>() { return 0; }

  //
  // UNIT TESTS
  //

  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( MultiVector, NonMemberConstructors, LO, GO, Scalar , Node )
  {
    typedef Tpetra::Map<LO, GO, Node> map_type;
    typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;
    typedef Tpetra::Vector<Scalar,LO,GO,Node> V;
    constexpr bool debug = true;

    RCP<Teuchos::FancyOStream> outPtr = debug ?
      Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cerr)) :
      Teuchos::rcpFromRef (out);
    Teuchos::FancyOStream& myOut = *outPtr;

    myOut << "Test: MultiVector, NonMemberConstructors" << endl;
    Teuchos::OSTab tab0 (myOut);

    myOut << "Create a Map" << endl;
    auto comm = getDefaultComm ();
    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid ();
    const size_t numLocal = 13;
    const size_t numVecs  = 7;
    const GO indexBase = 0;
    RCP<const map_type> map =
      rcp (new map_type (INVALID, numLocal, indexBase, comm));

    myOut << "Create a MultiVector, and make sure that it has "
      "the right number of vectors (columns)" << endl;
    RCP<MV> mvec = Tpetra::createMultiVector<Scalar>(map,numVecs);
    TEST_EQUALITY(mvec->getNumVectors(), numVecs);

    myOut << "Create a Vector, and make sure that "
      "it has exactly one vector (column)" << endl;
    RCP<V> vec = Tpetra::createVector<Scalar>(map);
    TEST_EQUALITY_CONST(vec->getNumVectors(), 1);

    // Make sure that the test passed on all processes, not just Proc 0.
    int lclSuccess = success ? 1 : 0;
    int gblSuccess = 1;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
    TEST_ASSERT( gblSuccess == 1 );
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( MultiVector, basic, LO, GO, Scalar , Node )
  {
    using map_type = Tpetra::Map<LO, GO, Node>;
    using MV = Tpetra::MultiVector<Scalar, LO, GO, Node>;
    using vec_type = Tpetra::Vector<Scalar, LO, GO, Node>;
    typedef typename ScalarTraits<Scalar>::magnitudeType Magnitude;
    constexpr bool debug = true;

    RCP<Teuchos::FancyOStream> outPtr = debug ?
      Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cerr)) :
      Teuchos::rcpFromRef (out);
    Teuchos::FancyOStream& myOut = *outPtr;

    myOut << "Test: MultiVector, basic" << endl;
    Teuchos::OSTab tab0 (myOut);

    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid ();
    RCP<const Comm<int> > comm = getDefaultComm ();
    const int numImages = comm->getSize ();

    myOut << "Create Map" << endl;
    const size_t numLocal = 13;
    const size_t numVecs  = 7;
    const GO indexBase = 0;
    RCP<const map_type> map =
      rcp (new map_type (INVALID, numLocal, indexBase, comm));

    myOut << "Test MultiVector's & Vector's default constructors" << endl;
    {
      MV defaultConstructedMultiVector;
      auto dcmv_map = defaultConstructedMultiVector.getMap ();
      TEST_ASSERT( dcmv_map.get () != nullptr );
      if (dcmv_map.get () != nullptr) {
        TEST_EQUALITY( dcmv_map->getGlobalNumElements (),
                       Tpetra::global_size_t (0) );
      }
      vec_type defaultConstructedVector;
      auto dcv_map = defaultConstructedVector.getMap ();
      TEST_ASSERT( dcv_map.get () != nullptr );
      if (dcv_map.get () != nullptr) {
        TEST_EQUALITY( dcv_map->getGlobalNumElements (),
                       Tpetra::global_size_t (0) );
      }
    }

    myOut << "Test MultiVector's usual constructor" << endl;
    RCP<MV> mvec;
    TEST_NOTHROW( mvec = rcp (new MV (map, numVecs, true)) );
    if (mvec.is_null ()) {
      myOut << "MV constructor threw an exception: returning" << endl;
      return;
    }
    TEST_EQUALITY( mvec->getNumVectors(), numVecs );
    TEST_EQUALITY( mvec->getLocalLength(), numLocal );
    TEST_EQUALITY( mvec->getGlobalLength(), numImages*numLocal );

    myOut << "Test that all norms are zero" << endl;
    Array<Magnitude> norms(numVecs), zeros(numVecs);
    std::fill(zeros.begin(),zeros.end(),ScalarTraits<Magnitude>::zero());
    TEST_NOTHROW( mvec->norm2(norms) );
    TEST_COMPARE_FLOATING_ARRAYS(norms,zeros,ScalarTraits<Magnitude>::zero());
    TEST_NOTHROW( mvec->norm1(norms) );
    TEST_COMPARE_FLOATING_ARRAYS(norms,zeros,ScalarTraits<Magnitude>::zero());
    TEST_NOTHROW( mvec->normInf(norms) );
    TEST_COMPARE_FLOATING_ARRAYS(norms,zeros,ScalarTraits<Magnitude>::zero());
    // print it
    myOut << *mvec << endl;

    // Make sure that the test passed on all processes, not just Proc 0.
    int lclSuccess = success ? 1 : 0;
    int gblSuccess = 1;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
    TEST_ASSERT( gblSuccess == 1 );
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( MultiVector, large, LO, GO, Scalar , Node )
  {
    using map_type = Tpetra::Map<LO, GO, Node>;
    using MV = Tpetra::MultiVector<Scalar, LO, GO, Node>;
    using vec_type = Tpetra::Vector<Scalar, LO, GO, Node>;
    typedef typename ScalarTraits<Scalar>::magnitudeType Magnitude;
    constexpr bool debug = true;

    RCP<Teuchos::FancyOStream> outPtr = debug ?
      Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cerr)) :
      Teuchos::rcpFromRef (out);
    Teuchos::FancyOStream& myOut = *outPtr;

    myOut << "Test: MultiVector, basic" << endl;
    Teuchos::OSTab tab0 (myOut);

    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid ();
    RCP<const Comm<int> > comm = getDefaultComm ();
    const int numImages = comm->getSize ();

    myOut << "Create Map" << endl;
    const size_t numLocal = 15000;
    const size_t numVecs  = 10;
    const GO indexBase = 0;
    RCP<const map_type> map =
      rcp (new map_type (INVALID, numLocal, indexBase, comm));

    myOut << "Test MultiVector's & Vector's default constructors" << endl;
    {
      MV defaultConstructedMultiVector;
      auto dcmv_map = defaultConstructedMultiVector.getMap ();
      TEST_ASSERT( dcmv_map.get () != nullptr );
      if (dcmv_map.get () != nullptr) {
        TEST_EQUALITY( dcmv_map->getGlobalNumElements (),
                       Tpetra::global_size_t (0) );
      }
      vec_type defaultConstructedVector;
      auto dcv_map = defaultConstructedVector.getMap ();
      TEST_ASSERT( dcv_map.get () != nullptr );
      if (dcv_map.get () != nullptr) {
        TEST_EQUALITY( dcv_map->getGlobalNumElements (),
                       Tpetra::global_size_t (0) );
      }
    }

    myOut << "Test MultiVector's usual constructor" << endl;
    RCP<MV> mvec;
    TEST_NOTHROW( mvec = rcp (new MV (map, numVecs, true)) );
    if (mvec.is_null ()) {
      myOut << "MV constructor threw an exception: returning" << endl;
      return;
    }
    TEST_EQUALITY( mvec->getNumVectors(), numVecs );
    TEST_EQUALITY( mvec->getLocalLength(), numLocal );
    TEST_EQUALITY( mvec->getGlobalLength(), numImages*numLocal );

    myOut << "Test that all norms are zero" << endl;
    Array<Magnitude> norms(numVecs), zeros(numVecs);
    std::fill(zeros.begin(),zeros.end(),ScalarTraits<Magnitude>::zero());
    TEST_NOTHROW( mvec->norm2(norms) );
    TEST_COMPARE_FLOATING_ARRAYS(norms,zeros,ScalarTraits<Magnitude>::zero());
    TEST_NOTHROW( mvec->norm1(norms) );
    TEST_COMPARE_FLOATING_ARRAYS(norms,zeros,ScalarTraits<Magnitude>::zero());
    TEST_NOTHROW( mvec->normInf(norms) );
    TEST_COMPARE_FLOATING_ARRAYS(norms,zeros,ScalarTraits<Magnitude>::zero());
    // print it
    myOut << *mvec << endl;

    // Make sure that the test passed on all processes, not just Proc 0.
    int lclSuccess = success ? 1 : 0;
    int gblSuccess = 1;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
    TEST_ASSERT( gblSuccess == 1 );
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( MultiVector, BadConstLDA, LO, GO, Scalar , Node )
  {
    // numlocal > LDA
    // ergo, the arrayview doesn't contain enough data to specify the entries
    // also, if bounds checking is enabled, check that bad bounds are caught
    typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;

    out << "Test: MultiVector, BadConstLDA" << endl;
    Teuchos::OSTab tab0 (out);

    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    // get a comm and node
    RCP<const Comm<int> > comm = getDefaultComm();
    const size_t numLocal = 2;
    const size_t numVecs = 2;
    // multivector has two vectors, each proc having two values per vector
    RCP<const Map<LO,GO,Node> > map = createContigMapWithNode<LO,GO,Node>(INVALID,numLocal,comm);
    // we need 4 scalars to specify values on each proc
    Array<Scalar> values(4);
#ifdef HAVE_TPETRA_DEBUG
    typedef Tpetra::Vector<Scalar,LO,GO,Node>       V;
    // too small an ArrayView (less than 4 values) is met with an exception, if debugging is on
    TEST_THROW(MV mvec(map,values(0,3),2,numVecs), std::invalid_argument);
    // it could also be too small for the given LDA:
    TEST_THROW(MV mvec(map,values(),2+1,numVecs), std::invalid_argument);
    // too small for number of entries in a Vector
    TEST_THROW(V   vec(map,values(0,1)), std::invalid_argument);
#endif
    // LDA < numLocal throws an exception anytime
    TEST_THROW(MV mvec(map,values(0,4),1,numVecs), std::runtime_error);

    // Make sure that the test passed on all processes, not just Proc 0.
    int lclSuccess = success ? 1 : 0;
    int gblSuccess = 1;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
    TEST_ASSERT( gblSuccess == 1 );
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( MultiVector, NonContigView, LO, GO, Scalar , Node )
  {
    typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;
    typedef Tpetra::Vector<Scalar,LO,GO,Node> V;
    typedef typename ScalarTraits<Scalar>::magnitudeType Mag;

    out << "Test: MultiVector, NonContigView: Scalar="
        << Teuchos::TypeNameTraits<Scalar>::name() << endl;
    Teuchos::OSTab tab0 (out);

    const Mag tol = errorTolSlack * errorTolSlack * testingTol<Scalar>();   // extra slack on this test; dots() seem to be a little sensitive for single precision types
    out << "tol: " << tol << endl;

    const Mag M0  = ScalarTraits<Mag>::zero();
    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    // create a Map
    const size_t numLocal = 53; // making this larger reduces the change that A below will have no non-zero entries, i.e., that C = abs(A) is still equal to A (we assume it is not)
    const size_t numVecs = 7;
    RCP<const Map<LO,GO,Node> > map = createContigMapWithNode<LO,GO,Node>(INVALID,numLocal,comm);
    //
    // we will create a non-contig subview of the vector; un-viewed vectors should not be changed
    Tuple<size_t,4> inView1 = tuple<size_t>(1,4,3,2);
    Tuple<size_t,3> exView1 = tuple<size_t>(0,5,6);
    Tuple<size_t,4> inView2 = tuple<size_t>(6,0,4,3);
    Tuple<size_t,4> exView2 = tuple<size_t>(1,2,5,7);
    const size_t numView = 4;
    TEUCHOS_TEST_FOR_EXCEPTION(numView != as<size_t>(inView1.size()), std::logic_error, "Someone ruined a test invariant.");
    TEUCHOS_TEST_FOR_EXCEPTION(numView != as<size_t>(inView1.size()), std::logic_error, "Someone ruined a test invariant.");
    TEUCHOS_TEST_FOR_EXCEPTION(numView != as<size_t>(inView2.size()), std::logic_error, "Someone ruined a test invariant.");
    {
      // test dot, all norms, randomize
      MV mvOrig1(map,numVecs), mvOrig2(map,numVecs+1), mvWeights(map,numVecs);
      mvWeights.randomize();
      RCP<const MV> mvW1 = mvWeights.subView(tuple<size_t>(0));
      RCP<const MV> mvSubWeights = mvWeights.subView(inView1);
      mvOrig1.randomize();
      mvOrig2.randomize();
      //
      Array<Mag> nOrig2(numVecs), nOrig1(numVecs), nOrigI(numVecs);
      Array<Scalar> meansOrig(numVecs), dotsOrig(numView);
      mvOrig1.norm1(nOrig1());
      mvOrig1.norm2(nOrig2());
      mvOrig1.normInf(nOrigI());
      mvOrig1.meanValue(meansOrig());
      for (size_t j=0; j < numView; ++j) {
        RCP<const V> v1 = mvOrig1.getVector(inView1[j]),
          v2 = mvOrig2.getVector(inView2[j]);
        dotsOrig[j] = v1->dot(*v2);
      }
      // create the views, compute and test
      RCP<      MV> mvView1 = mvOrig1.subViewNonConst(inView1);
      TEST_ASSERT( ! mvView1.is_null() );
      if (! mvView1.is_null() ) {
        {
          auto mvView1_d = mvView1->getLocalViewDevice(Tpetra::Access::ReadOnly);
          auto mvOrig1_d = mvOrig1.getLocalViewDevice(Tpetra::Access::ReadOnly);
          TEST_ASSERT( mvView1_d.data() == mvOrig1_d.data() );
        }
        {
          auto mvView1_h = mvView1->getLocalViewHost(Tpetra::Access::ReadOnly);
          auto mvOrig1_h = mvOrig1.getLocalViewHost(Tpetra::Access::ReadOnly);
          TEST_ASSERT( mvView1_h.data() == mvOrig1_h.data() );
        }
      }
      RCP<const MV> mvView2 = mvOrig2.subView(inView2);
      TEST_ASSERT( ! mvView2.is_null() );
      if (! mvView2.is_null() ) {
        {
          auto mvView2_lcl = mvView2->getLocalViewDevice(Tpetra::Access::ReadOnly);
          auto mvOrig2_lcl = mvOrig2.getLocalViewDevice(Tpetra::Access::ReadOnly);
          TEST_ASSERT( mvView2_lcl.data() == mvOrig2_lcl.data() );
        }
        {
          auto mvView2_h = mvView2->getLocalViewHost(Tpetra::Access::ReadOnly);
          auto mvOrig2_h = mvOrig2.getLocalViewHost(Tpetra::Access::ReadOnly);
          TEST_ASSERT( mvView2_h.data() == mvOrig2_h.data() );
        }
      }
      Array<Mag> nView2(numView), nView1(numView), nViewI(numView);
      Array<Scalar> meansView(numView), dotsView(numView);
      mvView1->norm1(nView1());
      mvView1->norm2(nView2());
      mvView1->normInf(nViewI());
      mvView1->meanValue(meansView());
      mvView1->dot( *mvView2, dotsView() );

      for (size_t j=0; j < numView; ++j) {
        TEST_FLOATING_EQUALITY(nOrig1[inView1[j]],  nView1[j],  tol);
      }
      for (size_t j=0; j < numView; ++j) {
        TEST_FLOATING_EQUALITY(nOrig2[inView1[j]],  nView2[j],  tol);
      }
      for (size_t j=0; j < numView; ++j) {
        TEST_FLOATING_EQUALITY(nOrigI[inView1[j]],  nViewI[j],  tol);
      }
      for (size_t j=0; j < numView; ++j) {
        TEST_FLOATING_EQUALITY(meansOrig[inView1[j]], meansView[j], tol);
      }
      for (size_t j=0; j < numView; ++j) {
        TEST_FLOATING_EQUALITY(dotsOrig[j], dotsView[j], tol);
      }

      int lclSuccess = success ? 1 : 0;
      int gblSuccess = 0;
      using Teuchos::outArg;
      using Teuchos::reduceAll;
      using Teuchos::REDUCE_MIN;
      reduceAll(*comm, REDUCE_MIN, lclSuccess, outArg(gblSuccess));
      TEST_EQUALITY_CONST( gblSuccess, 1 );
      if (! success) {
        return;
      }

      // randomize the view, compute view one-norms, test difference
      mvView2 = Teuchos::null;
      mvView1->randomize();
      Array<Mag> nView1_aft(numView);
      mvView1->norm1(nView1_aft());
      for (size_t j=0; j < numView; ++j) {
        TEST_INEQUALITY(nView1[j], nView1_aft[j]);
      }
      // release the view, test that viewed columns changed, others didn't
      mvView1 = Teuchos::null;
      Array<Mag> nOrig1_aft(numVecs);
      mvOrig1.norm1(nOrig1_aft());
      for (size_t j=0; j < as<size_t>(inView1.size()); ++j) {
        TEST_INEQUALITY(nOrig1[inView1[j]], nOrig1_aft[inView1[j]]);
      }
      for (size_t j=0; j < as<size_t>(exView1.size()); ++j) {
        TEST_FLOATING_EQUALITY(nOrig1[exView1[j]], nOrig1_aft[exView1[j]], tol);
      }
    }
    {
      MV mvOrigA(map,numVecs), mvOrigB(map,numVecs), mvOrigC(map,numVecs+1);
      // we don't know what the distribution is, so that we don't know that abs(A) != A
      // therefore, do some manipulation
      mvOrigA.randomize();
      mvOrigB.randomize();
      // A = A - B, then new (indendent) values for B
      mvOrigA.update(as<Scalar>(-1),mvOrigB, as<Scalar>(1));
      mvOrigB.randomize();
      mvOrigC.randomize();
      Array<Mag> nrmOrigA(numVecs), nrmOrigB(numVecs), nrmOrigC(numVecs+1);
      mvOrigA.norm2(nrmOrigA());
      mvOrigB.norm2(nrmOrigB());
      mvOrigC.norm2(nrmOrigC());
      RCP<MV> mvViewA = mvOrigA.subViewNonConst(inView1);
      RCP<MV> mvViewB = mvOrigB.subViewNonConst(inView1);
      RCP<MV> mvViewC = mvOrigC.subViewNonConst(inView2);
      // set C = abs(A)
      {
        Array<Scalar> mnA_bef(inView1.size()), mnC_bef(inView1.size()),
                      mnA_aft(inView1.size()), mnC_aft(inView1.size());
        mvViewA->meanValue(mnA_bef());
        mvViewC->meanValue(mnC_bef());
        mvViewC->abs(*mvViewA);
        mvViewA->meanValue(mnA_aft());
        mvViewC->meanValue(mnC_aft());
        for (size_t j=0; j < as<size_t>(inView1.size()); ++j) {
          TEST_FLOATING_EQUALITY(mnA_bef[j], mnA_aft[j], tol);
          TEST_INEQUALITY(mnC_bef[j], mnC_aft[j]);
        }
      }
      // then set A = B = C
      // good excuse for some double views
      // use full views of C and B for this, check means before and after
      // to make sure that only A and B change.
      {
        Array<Scalar> A_bef(inView1.size()), B_bef(inView1.size()), C_bef(inView2.size());
        mvViewA->meanValue(A_bef());
        mvViewB->meanValue(B_bef());
        mvViewC->meanValue(C_bef());
        RCP<MV> doubleViewA = mvViewA->subViewNonConst(Range1D(0,inView1.size()-1));
        RCP<MV> doubleViewB = mvViewB->subViewNonConst(Range1D(0,inView1.size()-1));
        RCP<const MV> doubleViewC = mvViewC->subView(Range1D(0,inView1.size()-1));
        //(*doubleViewA) = (*doubleViewB) = (*doubleViewC);
        deep_copy((*doubleViewB),(*doubleViewC));
        deep_copy((*doubleViewA),(*doubleViewB));
        doubleViewA = Teuchos::null;
        doubleViewB = Teuchos::null;
        doubleViewC = Teuchos::null;
        Array<Scalar> A_aft(inView1.size()), B_aft(inView1.size()), C_aft(inView2.size());
        mvViewA->meanValue(A_aft());
        mvViewB->meanValue(B_aft());
        mvViewC->meanValue(C_aft());
        for (size_t j=0; j < as<size_t>(inView1.size()); ++j) {
          TEST_FLOATING_EQUALITY(C_bef[j], C_aft[j], tol);
          TEST_FLOATING_EQUALITY(C_bef[j], B_aft[j], tol);
          TEST_FLOATING_EQUALITY(C_bef[j], A_aft[j], tol);
          TEST_INEQUALITY(A_bef[j], A_aft[j]);
          TEST_INEQUALITY(B_bef[j], B_aft[j]);
        }
      }
      {
        TEUCHOS_TEST_FOR_EXCEPTION(inView1.size() != 4, std::logic_error, "Someone ruined a test invariant.");
        Tuple<size_t,4> reorder = tuple<size_t>(3,1,0,2);
        RCP<MV> dvA = mvViewA->subViewNonConst(reorder);
        RCP<MV> dvB = mvViewB->subViewNonConst(reorder);
        RCP<MV> dvC = mvViewC->subViewNonConst(reorder);
        // C == B == A
        //   C *= 2                ->  C == 2*A == 2*B            scale(alpha)
        dvC->scale( as<Scalar>(2) );
        //   A = -C + 2*A          ->  C == 2*B, A == 0           update(alpha,mv,beta)
        dvA->update(as<Scalar>(-1),*dvC, as<Scalar>(2));
        //   C = 2*A + 2*B - .5*C ->   C == B, A == 0,            update(alpha,mv,beta,mv,gamma)
        dvC->update(as<Scalar>(2),*dvA, as<Scalar>(2), *dvB, as<Scalar>(-.5));
        //   B = 0.5              ->   B = 0.5, A == 0,           putScalar(alpha)
        dvB->putScalar( as<Scalar>(0.5) );
        //   C.recip(B)           ->   C = 2, B == 0.5, A == 0,   reciprocal(mv)
        dvC->reciprocal(*dvB);
        //   B = C/2              ->   A == 0, B == 1, C == 2
        dvB->scale(as<Mag>(0.5),*dvC);
        dvA = Teuchos::null;
        dvB = Teuchos::null;
        dvC = Teuchos::null;
        Array<Mag> nrmA(4), nrmB(4), nrmC(4);
        mvViewA->norm1(nrmA()); // norm1(0)   = 0
        mvViewB->norm1(nrmB()); // norm1(1.0) = N
        mvViewC->norm1(nrmC()); // norm1(2.0) = 2 * N
        const Mag  OneN = as<Mag>(mvViewA->getGlobalLength());
        const Mag  TwoN = OneN + OneN;
        for (size_t j=0; j < 4; ++j) {
          TEST_FLOATING_EQUALITY( nrmA[j],    M0, tol );
          TEST_FLOATING_EQUALITY( nrmB[j],  OneN, tol );
          TEST_FLOATING_EQUALITY( nrmC[j],  TwoN, tol );
        }
      }
      // done with these views; clear them, ensure that only the viewed
      // vectors changed in the original multivectors
      mvViewA = Teuchos::null;
      mvViewB = Teuchos::null;
      mvViewC = Teuchos::null;
      Array<Mag> nrmOrigA_aft(numVecs), nrmOrigB_aft(numVecs), nrmOrigC_aft(numVecs+1);
      mvOrigA.norm2(nrmOrigA_aft());
      mvOrigB.norm2(nrmOrigB_aft());
      mvOrigC.norm2(nrmOrigC_aft());
      for (size_t j=0; j < as<size_t>(inView1.size()); ++j) {
        TEST_INEQUALITY(nrmOrigA[inView1[j]], nrmOrigA_aft[inView1[j]]);
        TEST_INEQUALITY(nrmOrigB[inView1[j]], nrmOrigB_aft[inView1[j]]);
        TEST_INEQUALITY(nrmOrigC[inView2[j]], nrmOrigC_aft[inView2[j]]);
      }
      for (size_t j=0; j < as<size_t>(exView1.size()); ++j) {
        TEST_FLOATING_EQUALITY(nrmOrigA[exView1[j]], nrmOrigA_aft[exView1[j]], tol);
        TEST_FLOATING_EQUALITY(nrmOrigB[exView1[j]], nrmOrigB_aft[exView1[j]], tol);
      }
      for (size_t j=0; j < as<size_t>(exView1.size()); ++j) {
        TEST_FLOATING_EQUALITY(nrmOrigC[exView2[j]], nrmOrigC_aft[exView2[j]], tol);
      }
    }

    // Make sure that the test passed on all processes, not just Proc 0.
    int lclSuccess = success ? 1 : 0;
    int gblSuccess = 1;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
    TEST_ASSERT( gblSuccess == 1 );
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( MultiVector, Describable, LO , GO , Scalar, Node )
  {
    typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;

    out << "Test: MultiVector, Describable" << endl;
    Teuchos::OSTab tab0 (out);

    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    const int myImageID = comm->getRank();
    // create Map
    RCP<const Map<LO,GO,Node> > map = createContigMapWithNode<LO,GO,Node>(INVALID,3,comm);
    // test labeling
    const string lbl("mvecA");
    MV mvecA(map,2);
    string desc1 = mvecA.description();
    if (myImageID==0) out << desc1 << endl;
    mvecA.setObjectLabel(lbl);
    string desc2 = mvecA.description();
    if (myImageID==0) out << desc2 << endl;
    if (myImageID==0) {
      TEST_EQUALITY( mvecA.getObjectLabel(), lbl );
    }
    // test describing at different verbosity levels
    if (myImageID==0) out << "Describing with verbosity VERB_DEFAULT..." << endl;
    mvecA.describe(out);
    comm->barrier();
    comm->barrier();
    if (myImageID==0) out << "Describing with verbosity VERB_NONE..." << endl;
    mvecA.describe(out,VERB_NONE);
    comm->barrier();
    comm->barrier();
    if (myImageID==0) out << "Describing with verbosity VERB_LOW..." << endl;
    mvecA.describe(out,VERB_LOW);
    comm->barrier();
    comm->barrier();
    if (myImageID==0) out << "Describing with verbosity VERB_MEDIUM..." << endl;
    mvecA.describe(out,VERB_MEDIUM);
    comm->barrier();
    comm->barrier();
    if (myImageID==0) out << "Describing with verbosity VERB_HIGH..." << endl;
    mvecA.describe(out,VERB_HIGH);
    comm->barrier();
    comm->barrier();
    if (myImageID==0) out << "Describing with verbosity VERB_EXTREME..." << endl;
    mvecA.describe(out,VERB_EXTREME);
    comm->barrier();
    comm->barrier();
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( MultiVector, BadMultiply, LO , GO , Scalar , Node )
  {
    // mfh 05 May 2016: Tpetra::MultiVector::multiply only checks
    // local dimensions in a debug build.
#ifdef HAVE_TPETRA_DEBUG
    typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;

    out << "Test: MultiVector, BadMultiply" << endl;
    Teuchos::OSTab tab0 (out);

    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    const Scalar S1 = ScalarTraits<Scalar>::one(),
                 S0 = ScalarTraits<Scalar>::zero();
    // case 1: C(local) = A^X(local) * B^X(local)  : four of these
    {
      // create local Maps
      RCP<const Map<LO,GO,Node> > map3l = createLocalMapWithNode<LO,GO,Node>(3,comm),
                                  map2l = createLocalMapWithNode<LO,GO,Node>(2,comm);
      MV mvecA(map3l,2),
         mvecB(map2l,3),
         mvecD(map2l,2);
      // failures, 8 combinations:
      // [NTC],[NTC]: A,B don't match
      // [NTC],[NTC]: C doesn't match A,B
      TEST_THROW( mvecD.multiply(NO_TRANS  ,NO_TRANS  ,S1,mvecA,mvecA,S0), std::runtime_error);   // 2x2: 3x2 x 3x2
      TEST_THROW( mvecD.multiply(NO_TRANS  ,CONJ_TRANS,S1,mvecA,mvecB,S0), std::runtime_error);   // 2x2: 3x2 x 3x2
      TEST_THROW( mvecD.multiply(CONJ_TRANS,NO_TRANS  ,S1,mvecB,mvecA,S0), std::runtime_error);   // 2x2: 3x2 x 3x2
      TEST_THROW( mvecD.multiply(CONJ_TRANS,CONJ_TRANS,S1,mvecB,mvecB,S0), std::runtime_error);   // 2x2: 3x2 x 3x2
      TEST_THROW( mvecD.multiply(NO_TRANS  ,NO_TRANS  ,S1,mvecA,mvecB,S0), std::runtime_error);   // 2x2: 3x2 x 2x3
      TEST_THROW( mvecD.multiply(NO_TRANS  ,CONJ_TRANS,S1,mvecA,mvecA,S0), std::runtime_error);   // 2x2: 3x2 x 2x3
      TEST_THROW( mvecD.multiply(CONJ_TRANS,NO_TRANS  ,S1,mvecB,mvecB,S0), std::runtime_error);   // 2x2: 3x2 x 2x3
      TEST_THROW( mvecD.multiply(CONJ_TRANS,CONJ_TRANS,S1,mvecB,mvecA,S0), std::runtime_error);   // 2x2: 3x2 x 2x3
    }
    // case 2: C(local) = A^T(distr) * B  (distr)  : one of these
    {
      RCP<const Map<LO,GO,Node> > map3n = createContigMapWithNode<LO,GO,Node>(INVALID,3,comm);
      RCP<const Map<LO,GO,Node> > map2n = createContigMapWithNode<LO,GO,Node>(INVALID,2,comm);
      RCP<const Map<LO,GO,Node> > map2l = createLocalMapWithNode<LO,GO,Node>(2,comm),
                                  map3l = createLocalMapWithNode<LO,GO,Node>(3,comm);
      MV mv3nx2(map3n,2),
         mv2nx2(map2n,2),
         mv2lx2(map2l,2),
         mv2lx3(map2l,3),
         mv3lx2(map3l,2),
         mv3lx3(map3l,3);
      // non-matching input lengths
      TEST_THROW( mv2lx2.multiply(CONJ_TRANS,NO_TRANS,S1,mv3nx2,mv2nx2,S0), std::runtime_error);   // (2 x 3n) x (2n x 2) not compat
      TEST_THROW( mv2lx2.multiply(CONJ_TRANS,NO_TRANS,S1,mv2nx2,mv3nx2,S0), std::runtime_error);   // (2 x 2n) x (3n x 2) not compat
      // non-matching output size
      TEST_THROW( mv3lx3.multiply(CONJ_TRANS,NO_TRANS,S1,mv3nx2,mv3nx2,S0), std::runtime_error);   // (2 x 3n) x (3n x 2) doesn't fit 3x3
      TEST_THROW( mv3lx2.multiply(CONJ_TRANS,NO_TRANS,S1,mv3nx2,mv3nx2,S0), std::runtime_error);   // (2 x 3n) x (3n x 2) doesn't fit 3x2
      TEST_THROW( mv2lx3.multiply(CONJ_TRANS,NO_TRANS,S1,mv3nx2,mv3nx2,S0), std::runtime_error);   // (2 x 3n) x (3n x 2) doesn't fit 2x3
    }
    // case 3: C(distr) = A  (distr) * B^X(local)  : two of these
    {
      RCP<const Map<LO,GO,Node> > map3n = createContigMapWithNode<LO,GO,Node>(INVALID,3,comm);
      RCP<const Map<LO,GO,Node> > map2n = createContigMapWithNode<LO,GO,Node>(INVALID,2,comm);
      RCP<const Map<LO,GO,Node> > map2l = createLocalMapWithNode<LO,GO,Node>(2,comm),
                                  map3l = createLocalMapWithNode<LO,GO,Node>(3,comm);
      MV mv3nx2(map3n,2),
         mv2nx2(map2n,2),
         mv2x3(map2l,3),
         mv3x2(map3l,2);
      // non-matching input lengths
      TEST_THROW( mv3nx2.multiply(NO_TRANS,CONJ_TRANS,S1,mv3nx2,mv2x3,S0), std::runtime_error);   // (3n x 2) x (3 x 2) (trans) not compat
      TEST_THROW( mv3nx2.multiply(NO_TRANS,NO_TRANS  ,S1,mv3nx2,mv3x2,S0), std::runtime_error);   // (3n x 2) x (3 x 2) (nontrans) not compat
      // non-matching output sizes
      TEST_THROW( mv3nx2.multiply(NO_TRANS,CONJ_TRANS,S1,mv3nx2,mv3x2,S0), std::runtime_error);   // (3n x 2) x (2 x 3) doesn't fit 3nx2
      TEST_THROW( mv3nx2.multiply(NO_TRANS,NO_TRANS  ,S1,mv3nx2,mv2x3,S0), std::runtime_error);   // (3n x 2) x (2 x 3) doesn't fit 3nx2
    }

    // Make sure that the test passed on all processes, not just Proc 0.
    int lclSuccess = success ? 1 : 0;
    int gblSuccess = 1;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
    TEST_ASSERT( gblSuccess == 1 );
#endif // HAVE_TPETRA_DEBUG
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( MultiVector, Multiply, LO , GO , Scalar , Node )
  {
    using Teuchos::View;
    typedef typename ScalarTraits<Scalar>::magnitudeType Mag;
    typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;
    int lclSuccess = 1;
    int gblSuccess = 0;

    out << "Test: MultiVector, Multiply" << endl;
    Teuchos::OSTab tab0 (out);

    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    // get a comm and node
    RCP<const Comm<int> > comm = getDefaultComm();
    const int numImages = comm->getSize();
    // create a Map
    RCP<const Map<LO,GO,Node> > map3n = createContigMapWithNode<LO,GO,Node>(INVALID,3,comm);
    //RCP<const Map<LO,GO,Node> > map2n = createContigMapWithNode<LO,GO,Node>(INVALID,2,comm);
    RCP<const Map<LO,GO,Node> > lmap3 = createLocalMapWithNode<LO,GO,Node>(3,comm);
    RCP<const Map<LO,GO,Node> > lmap2 = createLocalMapWithNode<LO,GO,Node>(2,comm);
    const Scalar S1 = ScalarTraits<Scalar>::one();
    const Scalar S0 = ScalarTraits<Scalar>::zero();
    const Mag    M0 = ScalarTraits<Mag>::zero();
    // case 2: C(local) = A^T(distr) * B  (distr)  : one of these
    {
      MV mv3nx2(map3n,2),
         mv3nx3(map3n,3),
         // locals
         //mv2x2(lmap2,2),
         mv2x3(lmap2,3),
         mv3x2(lmap3,2);
         //mv3x3(lmap3,3);
      // fill multivectors with ones
      mv3nx3.putScalar(ScalarTraits<Scalar>::one());
      mv3nx2.putScalar(ScalarTraits<Scalar>::one());
      // fill expected answers Array

      Teuchos::Array<Scalar> check(9,3*numImages);
      // test
      //mv2x2.multiply(CONJ_TRANS,NO_TRANS,S1,mv3nx2,mv3nx2,S0);
      //{ auto tmpView = mv2x2.get1dView(); TEST_COMPARE_FLOATING_ARRAYS(tmpView,check(0,tmpView.size()),M0); }
      mv2x3.multiply(CONJ_TRANS,NO_TRANS,S1,mv3nx2,mv3nx3,S0);
      { auto tmpView = mv2x3.get1dView(); TEST_COMPARE_FLOATING_ARRAYS(tmpView,check(0,tmpView.size()),M0); }
      mv3x2.multiply(CONJ_TRANS,NO_TRANS,S1,mv3nx3,mv3nx2,S0);
      { auto tmpView = mv3x2.get1dView(); TEST_COMPARE_FLOATING_ARRAYS(tmpView,check(0,tmpView.size()),M0);}
      //mv3x3.multiply(CONJ_TRANS,NO_TRANS,S1,mv3nx3,mv3nx3,S0);
      //{ auto tmpView = mv3x3.get1dView(); TEST_COMPARE_FLOATING_ARRAYS(tmpView,check(0,tmpView.size()),M0);}
    }
    lclSuccess = success ? 1 : 0;
    gblSuccess = 0;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess,
                         outArg (gblSuccess));
    TEST_ASSERT( gblSuccess == 1 );
  }

//
// INSTANTIATIONS
//

#define UNIT_TEST_GROUP_BASE( SCALAR, LO, GO, NODE ) \
      TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT( MultiVector, Multiply          , LO, GO, SCALAR, NODE )

#ifdef KOKKOS_ENABLE_OPENMP
  // Add special test for OpenMP
  #define UNIT_TEST_GROUP( SCALAR, LO, GO, NODE ) \
    UNIT_TEST_GROUP_BASE( SCALAR, LO, GO, NODE ) \
    TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT( MultiVector, OpenMP_ThreadedSum, LO, GO, SCALAR, NODE )
#else
  #define UNIT_TEST_GROUP( SCALAR, LO, GO, NODE ) \
    UNIT_TEST_GROUP_BASE( SCALAR, LO, GO, NODE )
#endif






  typedef Tpetra::Map<>::local_ordinal_type default_local_ordinal_type;
  typedef Tpetra::Map<>::global_ordinal_type default_global_ordinal_type;

#if defined(HAVE_TEUCHOS_COMPLEX) && defined(HAVE_TPETRA_INST_COMPLEX_FLOAT)
#  define TPETRA_MULTIVECTOR_COMPLEX_FLOAT_DOT_TEST( NODE ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT( MultiVector, ComplexDotOneColumn, float, default_local_ordinal_type, default_global_ordinal_type, NODE )
#else
#  define TPETRA_MULTIVECTOR_COMPLEX_FLOAT_DOT_TEST( NODE )
#endif // defined(HAVE_TEUCHOS_COMPLEX) && defined(HAVE_TPETRA_INST_COMPLEX_FLOAT)

#if defined(HAVE_TEUCHOS_COMPLEX) && defined(HAVE_TPETRA_INST_COMPLEX_DOUBLE)
#  define TPETRA_MULTIVECTOR_COMPLEX_DOUBLE_DOT_TEST( NODE ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT( MultiVector, ComplexDotOneColumn, double, default_local_ordinal_type, default_global_ordinal_type, NODE )
#else
#  define TPETRA_MULTIVECTOR_COMPLEX_DOUBLE_DOT_TEST( NODE )
#endif // defined(HAVE_TEUCHOS_COMPLEX) && defined(HAVE_TPETRA_INST_COMPLEX_DOUBLE)



#define VIEWMODETEST(NODE) \

  TPETRA_ETI_MANGLING_TYPEDEFS()

  TPETRA_INSTANTIATE_N( TPETRA_MULTIVECTOR_COMPLEX_FLOAT_DOT_TEST )

  TPETRA_INSTANTIATE_N( TPETRA_MULTIVECTOR_COMPLEX_DOUBLE_DOT_TEST )

  TPETRA_INSTANTIATE_TESTMV( UNIT_TEST_GROUP )

}
