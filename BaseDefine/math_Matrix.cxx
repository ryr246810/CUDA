// Copyright (c) 1997-1999 Matra Datavision
// Copyright (c) 1999-2014 OPEN CASCADE SAS
//
// This file is part of Open CASCADE Technology software library.
//
// This library is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License version 2.1 as published
// by the Free Software Foundation, with special exception defined in the file
// OCCT_LGPL_EXCEPTION.txt. Consult the file LICENSE_LGPL_21.txt included in OCCT
// distribution for complete text of the license and disclaimer of any warranty.
//
// Alternatively, this file may be used under the terms of Open CASCADE
// commercial license or contractual agreement.

#include <math_Matrix.hxx>
#include <math_Vector.hxx>
#include <stdlib.h>
#include <cmath>

void math_Matrix::SetLowerRow(const Standard_Integer LowerRow) {
  
  Array.SetLowerRow(LowerRow);
  Standard_Integer Rows = RowNumber();
  LowerRowIndex = LowerRow;
  UpperRowIndex = LowerRowIndex + Rows - 1;
}

void math_Matrix::SetLowerCol(const Standard_Integer LowerCol) {
  
  Array.SetLowerCol(LowerCol);
  Standard_Integer Cols = ColNumber();
  LowerColIndex = LowerCol;
  UpperColIndex = LowerColIndex + Cols - 1;
}

math_Matrix::math_Matrix (const Standard_Integer LowerRow,
			  const Standard_Integer UpperRow,
			  const Standard_Integer LowerCol,
			  const Standard_Integer UpperCol): 
			  
			  LowerRowIndex(LowerRow),
			  UpperRowIndex(UpperRow),
			  LowerColIndex(LowerCol),
			  UpperColIndex(UpperCol),
			  Array(LowerRow, UpperRow,
				LowerCol, UpperCol) 
{
  if((LowerRow > UpperRow) ||
     (LowerCol > UpperCol)){
    cout<<"math_Matrix::math_Matrix----------------error-----------------range error"<<endl;
  }
}

math_Matrix::math_Matrix (const Standard_Integer LowerRow,
			  const Standard_Integer UpperRow,
			  const Standard_Integer LowerCol,
			  const Standard_Integer UpperCol,
			  const Standard_Real InitialValue): 
			  
			  LowerRowIndex(LowerRow),
			  UpperRowIndex(UpperRow),
			  LowerColIndex(LowerCol),
			  UpperColIndex(UpperCol),
			  Array(LowerRow, UpperRow,
				LowerCol, UpperCol) 
{
  if((LowerRow > UpperRow) ||
     (LowerCol > UpperCol)){
    cout<<"math_Matrix::math_Matrix----------------error-----------------range error"<<endl;
  }

  Array.Init(InitialValue);
}

math_Matrix::math_Matrix (const Standard_Address Tab,
			  const Standard_Integer LowerRow,
			  const Standard_Integer UpperRow,
			  const Standard_Integer LowerCol,
			  const Standard_Integer UpperCol) :
			  
			  LowerRowIndex(LowerRow),
			  UpperRowIndex(UpperRow),
			  LowerColIndex(LowerCol),
			  UpperColIndex(UpperCol),
			  Array(Tab, LowerRow, UpperRow, LowerCol, UpperCol) 
{ 
  if((LowerRow > UpperRow) ||
     (LowerCol > UpperCol)){
    cout<<"math_Matrix::math_Matrix----------------error-----------------range error"<<endl;
  }
}

void math_Matrix::Init(const Standard_Real InitialValue) 
{
  Array.Init(InitialValue);
}

math_Matrix::math_Matrix (const math_Matrix& Other): 

LowerRowIndex(Other.LowerRow()),
UpperRowIndex(Other.UpperRow()),
LowerColIndex(Other.LowerCol()),
UpperColIndex(Other.UpperCol()),
Array(Other.Array) 
{
}



math_Matrix math_Matrix::Divided (const Standard_Real Right) const 
{
  if(fabs(Right) <= RealEpsilon()){
    cout<<"math_Matrix::Divided-------------------error----------------divided by zero"<<endl;
  }

  math_Matrix temp = Multiplied(1./Right);
  return temp;
}


void math_Matrix::Transpose() 
{ 
  if(RowNumber() != ColNumber()){
    cout<<"math_Matrix::Transpose--------------------error---------------not square"<<endl;
  }

  Standard_Integer Row = LowerRowIndex;
  Standard_Integer Col = LowerColIndex;
  SetLowerCol(LowerRowIndex);
  Standard_Real Temp;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J = I; J <= UpperColIndex; J++) {
      Temp = Array(I, J);
      Array(I, J) = Array(J, I);
      Array(J, I) = Temp;
    }
  }
  SetLowerRow(Col);
  SetLowerCol(Row);
}

math_Matrix math_Matrix::Transposed() const 
{ 
  math_Matrix Result(LowerColIndex, UpperColIndex,
		     LowerRowIndex, UpperRowIndex);
  
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Result.Array(J, I) = Array(I, J);
    }
  }
  return Result;
}

void math_Matrix::Multiply (const Standard_Real Right) 
{ 
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Array(I, J) = Array(I, J) * Right;
    }
  }
}

math_Matrix math_Matrix::Multiplied (const Standard_Real Right) const
{ 
  math_Matrix Result(LowerRowIndex, UpperRowIndex, 
		     LowerColIndex, UpperColIndex);
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Result.Array(I, J) = Array(I, J) * Right;
    }
  }
  return Result;
}

math_Matrix math_Matrix::TMultiplied (const Standard_Real Right) const
{ 
  math_Matrix Result(LowerRowIndex, UpperRowIndex, 
		     LowerColIndex, UpperColIndex);
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Result.Array(I, J) = Array(I, J) * Right;
    }
  }
  return Result;
}



void math_Matrix::Divide (const Standard_Real Right) 
{ 
  if(fabs(Right) <= RealEpsilon()){
    cout<<"math_Matrix::Divide--------------------error-----------------------divided by zero"<<endl;
  }


  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Array(I, J) = Array(I, J) / Right;
    }
  }
}

void math_Matrix::Add (const math_Matrix& Right) 
{
  if((RowNumber() != Right.RowNumber()) ||
     (ColNumber() != Right.ColNumber())){
    cout<<"math_Matrix::Add----------------------error-------------------dimension error"<<endl;
  }


  Standard_Integer I2 = Right.LowerRowIndex;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    Standard_Integer J2 = Right.LowerColIndex;
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Array(I, J) = Array(I, J) + Right.Array(I2, J2);
      J2++;
    }
    I2++;
  }
}

void math_Matrix::Subtract (const math_Matrix& Right) 
{ 
  if((RowNumber() != Right.RowNumber()) ||
     (ColNumber() != Right.ColNumber())){
    cout<<"math_Matrix::Subtract----------------------error-------------------dimension error"<<endl;
  }

  Standard_Integer I2 = Right.LowerRowIndex;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    Standard_Integer J2 = Right.LowerColIndex;
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Array(I, J) = Array(I, J) - Right.Array(I2, J2);
      J2++;
    }
    I2++;
  }
}

void math_Matrix::Set(const Standard_Integer I1,const Standard_Integer I2,
		      const Standard_Integer J1,const Standard_Integer J2,
		      const math_Matrix& M) 
{
  if((I1 < LowerRowIndex)       || 
     (I2 > UpperRowIndex)       ||
     (J1 < LowerColIndex)       ||
     (J2 > UpperColIndex)       ||
     (I1 > I2) || (J1 > J2)     ||
     (I2-I1+1 != M.RowNumber()) ||
     (J2-J1+1 != M.ColNumber())){
    cout<<"math_Matrix::Set-------------------------error--------------dimension error"<<endl;
  }

  Standard_Integer II = M.LowerRow();
  for(Standard_Integer I = I1; I <= I2; I++) {
    Standard_Integer JJ = M.LowerCol();
    for(Standard_Integer J = J1; J <= J2; J++) {
      Array(I, J) = M.Array(II, JJ);
      JJ++;
    }
    II++;
  }
}         

void math_Matrix::SetRow (const Standard_Integer Row,
			  const math_Vector& V) 
{ 
  if((Row < LowerRowIndex) ||
     (Row > UpperRowIndex)){
    cout<<"math_Matrix::SetRow-------------------error---------------range error"<<endl;
  }
  if(ColNumber() != V.Length()){
    cout<<"math_Matrix::SetRow-------------------error---------------dimension error"<<endl;
  }


  Standard_Integer I = V.LowerIndex;
  for(Standard_Integer Index = LowerColIndex; Index <= UpperColIndex; Index++) {
    Array(Row, Index) = V.Array(I);
    I++;
  }
}

void math_Matrix::SetCol (const Standard_Integer Col,
			  const math_Vector& V) 
{ 
  if((Col < LowerColIndex) ||
     (Col > UpperColIndex)){
    cout<<"math_Matrix::SetCol-------------------error---------------range error"<<endl;
  }
  if(RowNumber() != V.Length()){
    cout<<"math_Matrix::SetCol-------------------error---------------dimension error"<<endl;
  }


  Standard_Integer I = V.LowerIndex;
  for(Standard_Integer Index = LowerRowIndex; Index <= UpperRowIndex; Index++) {
    Array(Index, Col) = V.Array(I);
    I++;
  }
}

void math_Matrix::SetDiag(const Standard_Real Value)
{ 
  if(RowNumber() != ColNumber()){
    cout<<"math_Matrix::SetDiag-----------------------------error--------------not square"<<endl;
  }

  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    Array(I, I) = Value;
  }
}
math_Vector math_Matrix::Row (const Standard_Integer Row)  const 
{ 
  
  math_Vector Result(LowerColIndex, UpperColIndex);
  
  for(Standard_Integer Index = LowerColIndex; Index <= UpperColIndex; Index++) {
    Result.Array(Index) = Array(Row, Index);
  }
  return Result;
}

math_Vector math_Matrix::Col (const Standard_Integer Col) const 
{ 
  
  math_Vector Result(LowerRowIndex, UpperRowIndex);
  
  for(Standard_Integer Index = LowerRowIndex; Index <= UpperRowIndex; Index++) {
    Result.Array(Index) = Array(Index, Col);
  }
  return Result;
}

void math_Matrix::SwapRow(const Standard_Integer Row1,
			  const Standard_Integer Row2) 
{
  if((Row1 < LowerRowIndex) ||
     (Row1 > UpperRowIndex) ||
     (Row2 < LowerRowIndex) ||
     (Row2 > UpperRowIndex)){
    cout<<"math_Matrix::SwapRow--------------------error---------------RangeError"<<endl;
  }

  math_Vector V1 = Row(Row1);
  math_Vector V2 = Row(Row2);
  SetRow(Row1,V2);
  SetRow(Row2,V1);
}

void math_Matrix::SwapCol(const Standard_Integer Col1,
			  const Standard_Integer Col2) 
{
  if((Col1 < LowerColIndex) ||
     (Col1 > UpperColIndex) ||
     (Col2 < LowerColIndex) ||
     (Col2 > UpperColIndex)){
    cout<<"math_Matrix::SwapCol--------------------error---------------------RangeError"<<endl;
  }

  math_Vector V1 = Col(Col1);
  math_Vector V2 = Col(Col2);
  SetCol(Col1,V2);
  SetCol(Col2,V1);
}



math_Matrix  math_Matrix::Multiplied (const math_Matrix& Right) const 
{
  if(ColNumber() != Right.RowNumber()){
    cout<<"math_Matrix::Multiplied-----------------------error--------------dimension error"<<endl;

  }


  math_Matrix Result(LowerRowIndex,       UpperRowIndex,
		     Right.LowerColIndex, Right.UpperColIndex);
  
  Standard_Real Som;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J2 = Right.LowerColIndex; J2 <= Right.UpperColIndex; J2++) {
      Som = 0.0;
      Standard_Integer I2 = Right.LowerRowIndex;
      for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
	Som = Som + Array(I, J) * Right.Array(I2, J2);
	I2++;
      }
      Result.Array(I, J2) = Som;
    }
  }
  return Result;
}


math_Matrix  math_Matrix::TMultiply (const math_Matrix& Right) const 
{ 
  if(RowNumber() != Right.RowNumber()){
    cout<<"math_Matrix::TMultiply-----------------------error--------------dimension error"<<endl;
  }

  math_Matrix Result(LowerColIndex,       UpperColIndex,
		     Right.LowerColIndex, Right.UpperColIndex);
  
  Standard_Real Som;
  for(Standard_Integer I = LowerColIndex; I <= UpperColIndex; I++) {
    for(Standard_Integer J2 = Right.LowerColIndex; J2 <= Right.UpperColIndex; J2++) {
      Som = 0.0;
      Standard_Integer I2 = Right.LowerRowIndex;
      for(Standard_Integer J = LowerRowIndex; J <= UpperRowIndex; J++) {
	Som = Som + Array(J, I) * Right.Array(I2, J2);
	I2++;
      }
      Result.Array(I, J2) = Som;
    }
  }
  return Result;
}


math_Matrix  math_Matrix::Added (const math_Matrix& Right) const 
{ 
  if((RowNumber() != Right.RowNumber()) ||
     (ColNumber() != Right.ColNumber())){
    cout<<"math_Matrix::Added-----------------------error----------------dimension error"<<endl;
  }

  math_Matrix Result(LowerRowIndex, UpperRowIndex, 
		     LowerColIndex, UpperColIndex);
  
  Standard_Integer I2 = Right.LowerRowIndex;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    Standard_Integer J2 = Right.LowerColIndex;
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Result.Array(I, J) = Array(I, J) + Right.Array(I2, J2);
      J2++;
    }
    I2++;
  }
  return Result;
}

math_Matrix  math_Matrix::Opposite () 
{ 
  
  math_Matrix Result(LowerRowIndex, UpperRowIndex, 
		     LowerColIndex, UpperColIndex);
  
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Result.Array(I, J) = - Array(I, J);
    }
  }
  return Result;
}

math_Matrix  math_Matrix::Subtracted (const math_Matrix& Right) const 
{ 
  if((RowNumber() != Right.RowNumber()) ||
     (ColNumber() != Right.ColNumber())){
    cout<<"math_Matrix::Subtracted ------------------------error---------------dimension error"<<endl;
  }

  math_Matrix Result(LowerRowIndex, UpperRowIndex, 
		     LowerColIndex, UpperColIndex);
  
  Standard_Integer I2 = Right.LowerRowIndex;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    Standard_Integer J2 = Right.LowerColIndex;
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Result.Array(I, J) = Array(I, J) - Right.Array(I2, J2);
      J2++;
    }
    I2++;
  }
  return Result;
}

void  math_Matrix::Multiply(const math_Vector&  Left, 
			    const math_Vector&  Right) 
{
  if((RowNumber() != Left.Length()) ||
     (ColNumber() != Right.Length())){
    cout<<"math_Matrix::Multiply----------------------error-----------------dimension error"<<endl;
  }
  
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Array(I, J) = Left.Array(I) * Right.Array(J);
    }
  }
}

void  math_Matrix::Multiply(const math_Matrix&  Left, 
			    const math_Matrix&  Right) 
{
  if((Left.ColNumber() != Right.RowNumber()) ||
     (RowNumber() != Left.RowNumber()) ||
     (ColNumber() != Right.ColNumber())){
    cout<<"math_Matrix::Multiply----------------------------error------------------dimension error"<<endl;
  }

  Standard_Real Som;
  Standard_Integer I1 = Left.LowerRowIndex;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    Standard_Integer J2 = Right.LowerColIndex;  
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Som = 0.0;
      Standard_Integer J1 = Left.LowerColIndex;
      Standard_Integer I2 = Right.LowerRowIndex;
      for(Standard_Integer K = Left.LowerColIndex; K <= Left.UpperColIndex; K++) {
	Som = Som + Left.Array(I1, J1) * Right.Array(I2, J2);
	J1++;
	I2++;
      }
      Array(I, J) = Som;
      J2++;
    }
    I1++;
  }
}

void math_Matrix::TMultiply(const math_Matrix& TLeft, 
			    const math_Matrix&  Right) 
{
  if((TLeft.RowNumber() != Right.RowNumber()) ||
     (RowNumber() != TLeft.ColNumber()) ||
     (ColNumber() != Right.ColNumber())){
    cout<<"math_Matrix::TMultiply---------------------------error----------------------dimension error"<<endl;
  }


  Standard_Real Som;
  Standard_Integer I1 = TLeft.LowerColIndex;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    Standard_Integer J2 = Right.LowerColIndex;  
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Som = 0.0;
      Standard_Integer J1 = TLeft.LowerRowIndex;
      Standard_Integer I2 = Right.LowerRowIndex;
      for(Standard_Integer K = TLeft.LowerRowIndex; K <= TLeft.UpperRowIndex; K++) {
	Som = Som + TLeft.Array(J1, I1) * Right.Array(I2, J2);
	J1++;
	I2++;
      }
      Array(I, J) = Som;
      J2++;
    }
    I1++;
  }
}

void  math_Matrix::Add (const math_Matrix&  Left, 
			const math_Matrix&  Right) 
{
  if((RowNumber() != Right.RowNumber()) ||
     (ColNumber() != Right.ColNumber()) ||
     (Right.RowNumber() != Left.RowNumber()) ||
     (Right.ColNumber() != Left.ColNumber())){
    cout<<"math_Matrix::Add--------------------------------error-----------------dimension error"<<endl;

  }

  Standard_Integer I1 = Left.LowerRowIndex;
  Standard_Integer I2 = Right.LowerRowIndex;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    Standard_Integer J1 = Left.LowerColIndex;
    Standard_Integer J2 = Right.LowerColIndex;
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Array(I, J) = Left.Array(I1, J1) + Right.Array(I2, J2);
      J1++;
      J2++;
    }
    I1++;
    I2++;
  }
}

void  math_Matrix::Subtract(const math_Matrix&  Left,
			    const math_Matrix&  Right) 
{
  if((RowNumber() != Right.RowNumber()) ||
     (ColNumber() != Right.ColNumber()) ||
     (Right.RowNumber() != Left.RowNumber()) ||
     (Right.ColNumber() != Left.ColNumber())){
    cout<<"math_Matrix::Subtract------------------------------error-----------------dimension error"<<endl;
  }
     

  Standard_Integer I1 = Left.LowerRowIndex;
  Standard_Integer I2 = Right.LowerRowIndex;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    Standard_Integer J1 = Left.LowerColIndex;
    Standard_Integer J2 = Right.LowerColIndex;
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Array(I, J) = Left.Array(I1, J1) - Right.Array(I2, J2);
      J1++;
      J2++;
    }
    I1++;
    I2++;
  }
}


void math_Matrix::Multiply(const math_Matrix& Right) 
{
  if(ColNumber() != Right.RowNumber()){
    cout<<"math_Matrix::Multiply-------------------error----------------dimension error"<<endl;
  }


  Standard_Real Som;
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J2 = Right.LowerColIndex; J2 <= Right.UpperColIndex; J2++) {
      Som = 0.0;
      Standard_Integer I2 = Right.LowerRowIndex;
      for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
	Som = Som + Array(I, J) * Right.Array(I2, J2);
	I2++;
      }
      Array(I, J2) = Som;
    }
  }
}



math_Vector math_Matrix::Multiplied(const math_Vector& Right)const
{
  if(ColNumber() != Right.Length()){
    cout<<"math_Matrix::Multiplied------------------error--------------dimension error"<<endl;
  }

  math_Vector Result(LowerRowIndex, UpperRowIndex);
  
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    Result.Array(I) = 0.0;
    Standard_Integer II = Right.LowerIndex;
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      Result.Array(I) = Result.Array(I) + Array(I, J) * Right.Array(II);
      II++;
    }
  }
  return Result;
}

math_Matrix& math_Matrix::Initialized(const math_Matrix& Other) 
{
  if((RowNumber() != Other.RowNumber()) ||
     (ColNumber() != Other.ColNumber())){
    cout<<"math_Matrix::Initialized-------------------error------------------dimension error"<<endl; 
  }


  (Other.Array).Copy(Array);
  return *this;
}




void math_Matrix::Dump(ostream& o)const

{
  o << "math_Matrix of RowNumber = " << RowNumber();
  o << " and ColNumber = " << ColNumber() << "\n";
  
  for(Standard_Integer I = LowerRowIndex; I <= UpperRowIndex; I++) {
    for(Standard_Integer J = LowerColIndex; J <= UpperColIndex; J++) {
      o << "math_Matrix ( " << I << ", " << J << " ) = ";
      o << Array(I, J) << "\n";
    }
  }
}

