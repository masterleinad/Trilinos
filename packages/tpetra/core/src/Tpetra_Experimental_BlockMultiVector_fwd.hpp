// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
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
// ************************************************************************
// @HEADER

#ifndef TPETRA_EXPERIMENTAL_BLOCKMULTIVECTOR_FWD_HPP
#define TPETRA_EXPERIMENTAL_BLOCKMULTIVECTOR_FWD_HPP

#include "Tpetra_Details_DefaultTypes.hpp"

/// \file Tpetra_Experimental_BlockMultiVector_fwd.hpp
/// \brief Forward declaration of Tpetra::Experimental::BlockMultiVector.

namespace Tpetra {
namespace Experimental {

/// \brief Implementation detail of Tpetra, to aid in deprecating
///   template parameters.
///
/// \warning This namespace is an implementation detail of Tpetra.  Do
///   <i>NOT</i> use it.  For any class CLASS in Tpetra::Experimental,
///   use the alias Tpetra::Experimental::CLASS, <i>NOT</i>
///   Tpetra::Experimental::Classes::CLASS.
namespace Classes {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <class SC, class LO, class GO, class NT> class BlockMultiVector;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace Classes

//! Alias for Tpetra::Experimental::Classes::BlockMultiVector.
template<class SC = ::Tpetra::Details::DefaultTypes::scalar_type,
         class LO = ::Tpetra::Details::DefaultTypes::local_ordinal_type,
         class GO = ::Tpetra::Details::DefaultTypes::global_ordinal_type,
         class NT = ::Tpetra::Details::DefaultTypes::node_type>
using BlockMultiVector = ::Tpetra::Experimental::Classes::BlockMultiVector<SC, LO, GO, NT>;

} // namespace Experimental
} // namespace Tpetra

#endif // TPETRA_EXPERIMENTAL_BLOCKMULTIVECTOR_FWD_HPP
