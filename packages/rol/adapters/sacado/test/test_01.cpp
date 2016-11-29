// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER


/*! \file  test_01.cpp
    \brief Test the NonlinearProblem interface with the Hock & Schittkowski
           problem set
*/

#define OPTIMIZATION_PROBLEM_REFACTOR 

#include "HS_ProblemFactory.hpp"

#include "ROL_OptimizationSolver.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_RandomVector.hpp"

#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include <iomanip>

typedef double RealT;

int main(int argc, char *argv[]) {

  using namespace ROL;
  using Teuchos::RCP;
  using Teuchos::rcp;
  
  typedef Teuchos::ParameterList PL;

  typedef Vector<RealT>              V;
  typedef PartitionedVector<RealT>   PV;
  typedef OptimizationProblem<RealT> OPT;
  typedef NonlinearProgram<RealT>    NLP;
  typedef AlgorithmState<RealT>      STATE;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = rcp(&std::cout, false);
  else
    outStream = rcp(&bhs, false);

  int errorFlag   = 0;
  int numProblems = 20;

  RealT errtol = 1e-5; // Require norm of computed solution error be less than this

  std::vector<int> passedTests;
  std::vector<int> failedTests;
  std::vector<std::string> converged;
  std::vector<std::string> stepname;

  HS::ProblemFactory<RealT> factory;

  RCP<NLP>  nlp;
  RCP<OPT>  opt;
  RCP<PL>   parlist = rcp( new PL() );

  std::string paramfile("hs_parameters.xml");
  Teuchos::updateParametersFromXmlFile(paramfile,parlist.ptr());

  bool verbose = parlist->get("Verbose",false);

  RCP<V>           x;
  RCP<const STATE> algo_state;

  bool problemSolved;

  // *** Test body.
  try {
 
    for(int n=1; n<=numProblems; ++n) {
      *outStream << "\nHock & Schittkowski Problem " << std::to_string(n) << std::endl;

      nlp = factory.getProblem(n);
      opt = nlp->getOptimizationProblem();
      EProblem problemType = opt->getProblemType();

      std::string str;

      switch( problemType ) {

        case TYPE_U:                                   
          str = parlist->get("Type-U Step","Trust Region");
        break;

        case TYPE_B:           
          str = parlist->get("Type-B Step","Trust Region");
        break;

        case TYPE_E:  
          str = parlist->get("Type-E Step","Composite Step");
        break;

        case TYPE_EB:
          str = parlist->get("Type-EB Step","Augmented Lagrangian");
        break;

        case TYPE_LAST:
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Error: Unsupported problem type!");
        break;
      }

      stepname.push_back(str);

      parlist->sublist("Step").set("Type",str);

      OptimizationSolver<RealT> solver( *opt, *parlist );

      if(verbose) {
        solver.solve(*outStream);
      }
      else {
        solver.solve();
      }

      algo_state = solver.getAlgorithmState();

      x = opt->getSolutionVector();       
     
      if( nlp->dimension_ci() > 0 ) { // Has slack variables, extract optimization vector

        PV xpv = Teuchos::dyn_cast<PV>(*x);
        RCP<V>  sol = xpv.get(0);
        
        problemSolved = nlp->foundAcceptableSolution( *sol, errtol );
  
      }
      else {
        problemSolved = nlp->foundAcceptableSolution( *x, errtol );
      }

      RealT tol = std::sqrt(ROL_EPSILON<RealT>()); 

      *outStream << "Target objective value   = " << nlp->getSolutionObjectiveValue()   << std::endl;
      *outStream << "Attained objective value = " << opt->getObjective()->value(*x,tol) << std::endl;

      if( problemSolved ) {
        *outStream << "Computed an acceptable solution" << std::endl;        
        passedTests.push_back(n);
        converged.push_back(std::to_string(algo_state->nfval));
      }
      else {
        *outStream << "Failed to converge" << std::endl;
        failedTests.push_back(n);
        converged.push_back("Failed");
      }

    } // end for loop over HS problems  


    *outStream << "\nTests passed: ";
    for( auto const& value : passedTests ) {
      *outStream << value << " ";
    }
    *outStream << "\nTests failed: ";
    for( auto const& value : failedTests ) {
      *outStream << value << " ";
    }

    *outStream << std::endl;

    if( passedTests.size() > failedTests.size() ) {
      *outStream << "Most tests passed." << std::endl; 
    }
    else {
      *outStream << "Most tests failed." << std::endl;
      errorFlag++;
    }
         
    *outStream << "\n\nPERFORMANCE SUMMARY:\n\n";
    *outStream << std::setw(16) << "H&S Problem #" 
               << std::setw(24) << "Step Used" 
               << std::setw(12) << "#fval" << std::endl;
    *outStream << std::string(52,'-') << std::endl;

    for(int n=0; n<numProblems; ++n) {
      *outStream << std::setw(16) << n+1
                 << std::setw(24) << stepname[n] 
                 << std::setw(12) << converged[n] << std::endl;
    }
   

  }
  catch (std::logic_error err) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;
} 

