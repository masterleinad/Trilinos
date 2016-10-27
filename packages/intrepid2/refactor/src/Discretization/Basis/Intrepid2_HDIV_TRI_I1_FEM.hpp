// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
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
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov), or
//                    Mauro Perego  (mperego@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file   Intrepid_HDIV_TRI_I1_FEM.hpp
    \brief  Header file for the Intrepid2::HDIV_TRI_I1_FEM class.
    \author Created by P. Bochev and D. Ridzal and K. Peterson.
 */

#ifndef INTREPID2_HDIV_TRI_I1_FEM_HPP
#define INTREPID2_HDIV_TRI_I1_FEM_HPP
#include "Intrepid2_Basis.hpp"

namespace Intrepid2 {
  
/** \class  Intrepid2::Basis_HDIV_TRI_I1_FEM
    \brief  Implementation of the default H(div)-compatible FEM basis of degree 1 on a Triangle cell.
  
            Implements Raviart-Thomas basis of degree 1 on the reference Triangle cell. The basis has
            cardinality 3 and spans an INCOMPLETE linear polynomial space. Basis functions are dual 
            to a unisolvent set of degrees-of-freedom (DoF) defined and enumerated as follows:
  
  \verbatim
  =========================================================================================================
  |         |           degree-of-freedom-tag table                    |                                  |
  |   DoF   |----------------------------------------------------------|       DoF definition             |
  | ordinal |  subc dim    | subc ordinal | subc DoF ord |subc num DoF |                                  |
  |=========|==============|==============|==============|=============|==================================|
  |    0    |       1      |       0      |       0      |      1      | L_0(u) = (u.n)(1/2,0)            |
  |---------|--------------|--------------|--------------|-------------|----------------------------------|
  |    1    |       1      |       1      |       0      |      1      | L_1(u) = (u.n)(1/2,1/2)          |
  |---------|--------------|--------------|--------------|-------------|----------------------------------|
  |    2    |       1      |       2      |       0      |      1      | L_2(u) = (u.n)(0,1/2)            |
  |=========|==============|==============|==============|=============|==================================|
  |   MAX   |  maxScDim=1  |  maxScOrd=2  |  maxDfOrd=0  |      -      |                                  |
  |=========|==============|==============|==============|=============|==================================|
  \endverbatim
  
    \remarks 
    \li     In the DOF functional \f${\bf n}=(t_2,-t_1)\f$ where \f${\bf t}=(t_1,t_2)\f$ 
            is the side (edge) tangent, i.e., the choice of normal direction is such that 
            the pair \f$({\bf n},{\bf t})\f$ is positively oriented. 
  
    \li     Direction of side tangents is determined by the vertex order of the sides in the
            cell topology and runs from side vertex 0 to side vertex 1, whereas their length is set 
            equal to the side length. For example, side 1 of all Triangle reference cells has vertex 
            order {1,2}, i.e., its tangent runs from vertex 1 of the reference Triangle to vertex 2 
            of that cell. On the reference Triangle the coordinates of these vertices are (1,0) and 
            (0,1), respectively. Therefore, the tangent to side 1 is (0,1)-(1,0) = (-1,1) and the 
            normal to that side is (1,1). Because its length already equals side length, no further 
            rescaling of the side tangent is needed.
        
    \li     The length of the side normal equals the length of the side. As a result, the 
            DoF functional is the value of the normal component of a vector field 
            at the side center times the side length. The resulting basis is equivalent to
            a basis defined by using the side flux as a DoF functional. Note that sides 0 and 2 of 
            reference Triangle<> cells have length 1 and side 1 has length Sqrt(2).
  
 */
  
  template<typename ExecSpaceType = void,
           typename outputValueType = double,
           typename pointValueType = double>
  class Basis_HDIV_TRI_I1_FEM: public Basis<ExecSpaceType,outputValueType,pointValueType> {
  public:
    typedef typename Basis<ExecSpaceType,outputValueType,pointValueType>::ordinal_type_array_1d_host ordinal_type_array_1d_host;
    typedef typename Basis<ExecSpaceType,outputValueType,pointValueType>::ordinal_type_array_2d_host ordinal_type_array_2d_host;
    typedef typename Basis<ExecSpaceType,outputValueType,pointValueType>::ordinal_type_array_3d_host ordinal_type_array_3d_host;

    template<EOperator opType>
    struct Serial {
      template<typename outputValueValueType, class ...outputValueProperties,
               typename inputPointValueType,  class ...inputPointProperties>
      KOKKOS_INLINE_FUNCTION
      static void
      getValues( /**/  Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
                 const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints );

    };


    template<typename outputValueViewType,
             typename inputPointViewType,
             EOperator opType>
    struct Functor {
      /**/  outputValueViewType _outputValues;
      const inputPointViewType  _inputPoints;

      KOKKOS_INLINE_FUNCTION
      Functor( /**/  outputValueViewType outputValues_,
               /**/  inputPointViewType  inputPoints_ )
        : _outputValues(outputValues_), _inputPoints(inputPoints_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const ordinal_type pt) const {
        switch (opType) {
        case OPERATOR_VALUE : {
          auto       output = Kokkos::subdynrankview( _outputValues, Kokkos::ALL(), pt, Kokkos::ALL() );
          const auto input  = Kokkos::subdynrankview( _inputPoints,                 pt, Kokkos::ALL() );
          Serial<opType>::getValues( output, input );
          break;
        }
        case OPERATOR_DIV : {
          auto       output = Kokkos::subdynrankview( _outputValues, Kokkos::ALL(), pt );
          const auto input  = Kokkos::subdynrankview( _inputPoints,                 pt, Kokkos::ALL() );
          Serial<opType>::getValues( output, input );
          break;
        }
        default: {
          INTREPID2_TEST_FOR_ABORT( opType != OPERATOR_VALUE &&
                                    opType != OPERATOR_DIV,
                                    ">>> ERROR: (Intrepid2::Basis_HDIV_TRI_I1_FEM::Serial::getValues) operator is not supported");
        }
        }
      }
    };
  
    class Internal {
    private:
      Basis_HDIV_TRI_I1_FEM *obj_;
      
    public:
      Internal(Basis_HDIV_TRI_I1_FEM *obj)
        : obj_(obj) {}
      
      /** \brief  Evaluation of a FEM basis on a <strong>reference Triangle</strong> cell.
          
          Returns values of <var>operatorType</var> acting on FEM basis functions for a set of
          points in the <strong>reference Triangle</strong> cell. For rank and dimensions of
          I/O array arguments see Section \ref basis_md_array_sec .
          
          \param  outputValues      [out] - variable rank array with the basis values
          \param  inputPoints       [in]  - rank-2 array (P,D) with the evaluation points
          \param  operatorType      [in]  - the operator acting on the basis functions
      */
      template<typename outputValueValueType, class ...outputValueProperties,
               typename inputPointValueType,  class ...inputPointProperties>
      void 
      getValues( /**/  Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
                 const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
                 const EOperator operatorType  = OPERATOR_VALUE ) const;

      /** \brief  Returns spatial locations (coordinates) of degrees of freedom on a
          <strong>reference Triangule</strong>.
          
          \param  DofCoords      [out] - array with the coordinates of degrees of freedom,
          dimensioned (F,D)
      */
      template<typename dofCoordValueType, class ...dofCoordProperties>
      void
      getDofCoords( Kokkos::DynRankView<dofCoordValueType,dofCoordProperties...> dofCoords ) const;
    };
    Internal impl_;
    
    /** \brief  Constructor.
     */
    Basis_HDIV_TRI_I1_FEM();
    Basis_HDIV_TRI_I1_FEM(const Basis_HDIV_TRI_I1_FEM &b)
      : Basis<ExecSpaceType,outputValueType,pointValueType>(b),
        impl_(this) {}

    Basis_HDIV_TRI_I1_FEM& operator=(const Basis_HDIV_TRI_I1_FEM &b) {
      if (this != &b) {
        Basis<ExecSpaceType,outputValueType,pointValueType>::operator= (b);
        // do not copy impl
      }
      return *this;
    }

    typedef typename Basis<ExecSpaceType,outputValueType,pointValueType>::outputViewType outputViewType;
    typedef typename Basis<ExecSpaceType,outputValueType,pointValueType>::pointViewType  pointViewType;
    typedef typename Basis<ExecSpaceType,outputValueType,pointValueType>::scalarViewType  scalarViewType;

    virtual
    void
    getValues( /**/  outputViewType outputValues,
               const pointViewType  inputPoints,
               const EOperator operatorType = OPERATOR_VALUE ) const {
      impl_.getValues( outputValues, inputPoints, operatorType );
    }

    virtual
    void
    getDofCoords( scalarViewType dofCoords ) const {
      impl_.getDofCoords( dofCoords );
    }

    virtual
    const char*
    getName() const {
      return "Intrepid2_HDIV_TRI_I1_FEM";
    }

    virtual
    bool
    requireOrientation() const {
      return true;
    }

  };

}// namespace Intrepid2

#include "Intrepid2_HDIV_TRI_I1_FEMDef.hpp"

#endif