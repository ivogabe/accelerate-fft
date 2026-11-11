{-# LANGUAGE GADTs               #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}
-- |
-- Module      : Data.Array.Accelerate.Math.FFT.LLVM.Native
-- Copyright   : [2017..2020] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Math.FFT.LLVM.Native (

  fftMany,
  fft1D,
  fft2D,
  fft3D,

) where

import Data.Array.Accelerate.Math.FFT.Mode
import Data.Array.Accelerate.Math.FFT.Type

import Data.Array.Accelerate.AST.Operation
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Type

import Data.Primitive.Vec

import Data.Array.Accelerate.LLVM.Native.Foreign
import Data.Array.Accelerate.LLVM.Native.Operation

import Prelude                                                      as P

fftMany :: Mode -> NumericR e -> ForeignAcc (Array DIM2 (Vec2 e) -> Array DIM2 (Vec2 e))
fftMany mode eR = case eR of
  NumericRfloat32 -> mkForeignFft mode dim2 eR kernelManyFloat
  NumericRfloat64 -> mkForeignFft mode dim2 eR kernelManyDouble

fft1D :: Mode -> NumericR e -> ForeignAcc (Array DIM1 (Vec2 e) -> Array DIM1 (Vec2 e))
fft1D mode eR = case eR of
  NumericRfloat32 -> mkForeignFft mode dim1 eR kernelDim1Float
  NumericRfloat64 -> mkForeignFft mode dim1 eR kernelDim1Double

fft2D :: Mode -> NumericR e -> ForeignAcc (Array DIM2 (Vec2 e) -> Array DIM2 (Vec2 e))
fft2D mode eR = case eR of
  NumericRfloat32 -> mkForeignFft mode dim2 eR kernelDim2Float
  NumericRfloat64 -> mkForeignFft mode dim2 eR kernelDim2Double

fft3D :: Mode -> NumericR e -> ForeignAcc (Array DIM3 (Vec2 e) -> Array DIM3 (Vec2 e))
fft3D mode eR = case eR of
  NumericRfloat32 -> mkForeignFft mode dim3 eR kernelDim3Float
  NumericRfloat64 -> mkForeignFft mode dim3 eR kernelDim3Double

mkForeignFft :: forall sh e. Mode -> ShapeR sh -> NumericR e -> Kernel sh e -> ForeignAcc (Array sh (Vec2 e) -> Array sh (Vec2 e))
mkForeignFft mode shr eR kernel =
  ForeignAcc name $ foreignCallMaplikeWithConst
    -- Sign
    (TupRsingle scalarTypeInt) (signOfMode mode)
    -- Types of input and output arrays
    shr (TupRsingle tp) (TupRsingle tp)
    -- Operation we want to call
    NotFusible $ NForeign name True kernel
  where
    name = "fft_dim" ++ show (rank shr) ++ "_" ++ nameNum
    nameNum = case eR of
      NumericRfloat32 -> "float"
      NumericRfloat64 -> "double"

    tp :: ScalarType (Vec2 e)
    tp = VectorScalarType $ VectorType 2 $ case eR of
      NumericRfloat32 -> singleType
      NumericRfloat64 -> singleType

type Kernel sh e = ForeignAfun (Var' Int -> In sh (Vec2 e) -> Out sh (Vec2 e) -> ())
foreign import ccall unsafe "&accelerate_fft_dim1_double" kernelDim1Double :: Kernel DIM1 Double
foreign import ccall unsafe "&accelerate_fft_dim1_float"  kernelDim1Float  :: Kernel DIM1 Float
foreign import ccall unsafe "&accelerate_fft_dim2_double" kernelDim2Double :: Kernel DIM2 Double
foreign import ccall unsafe "&accelerate_fft_dim2_float"  kernelDim2Float  :: Kernel DIM2 Float
foreign import ccall unsafe "&accelerate_fft_dim3_double" kernelDim3Double :: Kernel DIM3 Double
foreign import ccall unsafe "&accelerate_fft_dim3_float"  kernelDim3Float  :: Kernel DIM3 Float
foreign import ccall unsafe "&accelerate_fft_many_double" kernelManyDouble :: Kernel DIM2 Double
foreign import ccall unsafe "&accelerate_fft_many_float"  kernelManyFloat  :: Kernel DIM2 Float
