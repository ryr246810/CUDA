#include <BaseDataDefine.hxx>
#include <PortDataFunc.hxx>

#include <iostream>
using namespace std;

/*
  cout<<"theMaterialType \t=\t "<<theMaterialType<<endl;
  cout<<"MURPORT = "<<MURPORT<<endl;
  cout<<"OPENPORT = "<<OPENPORT<<endl;
  cout<<"INPUTPORT = "<<INPUTPORT<<endl;
  cout<<"PECPORT = "<<PECPORT<<endl;
//*/

bool 
IsOpenMurPortType(const Standard_Integer theMaterialType)
{
  bool result = false;

  if( (Standard_Integer)(theMaterialType & MURPORT) != 0 ){
    result = true;
  }
  return result;
}


bool 
IsInputMurPortType(const Standard_Integer theMaterialType)
{
  bool result = false;

  if( (Standard_Integer)(theMaterialType & INPUTMURPORT) != 0 ){
    result = true;
  }
  return result;
}


bool 
IsMurPortType(const Standard_Integer theMaterialType)
{
  bool result = false;

  if( ((Standard_Integer)(theMaterialType & MURPORT) != 0) ||
      ((Standard_Integer)(theMaterialType & INPUTMURPORT) != 0) ){
    result = true;
  }

  return result;
}


bool 
IsPECPortType(const Standard_Integer theMaterialType)
{
  bool result = false;
  if( (Standard_Integer)(theMaterialType & PECPORT) != 0 ){
    result = true;
  }
  return result;
}


bool 
IsPMLPortType(const Standard_Integer theMaterialType)
{
  bool result = false;

  if( ((Standard_Integer)(theMaterialType & OPENPORT) != 0) || 
      ((Standard_Integer)(theMaterialType & INPUTPORT) != 0) ){
    result = true;
  }

  return result;
}


bool 
IsInputPMLPortType(const Standard_Integer theMaterialType)
{
  bool result = false;

  if( ((Standard_Integer)(theMaterialType & INPUTPORT) != 0) ){
    result = true;
  }

  return result;
}



bool 
IsOnePortType(const Standard_Integer theMaterialType, 
	      Standard_Integer& thePortType)
{
  bool result = false;
  thePortType = 0;

  Standard_Integer refType = OPENPORT + INPUTPORT + PECPORT + MURPORT + INPUTMURPORT;

  if( (theMaterialType & refType) != 0 ){
    result = true;
    thePortType = theMaterialType & refType;
  }

  return result;
}



/*
bool 
IsOnePortType(const Standard_Integer theMaterialType, 
	      Standard_Integer& thePortType)
{
  bool result = false;
  thePortType = 0;


  if( ((Standard_Integer)(theMaterialType & OPENPORT) != 0) || 
      ((Standard_Integer)(theMaterialType & INPUTPORT) != 0) ||
      ((Standard_Integer)(theMaterialType & MURPORT) != 0) ||
      ((Standard_Integer)(theMaterialType & INPUTMURPORT) != 0) ||
      ((Standard_Integer)(theMaterialType & PECPORT) != 0)){

    result = true;

    thePortType = 
      (theMaterialType & OPENPORT) + 
      (theMaterialType & INPUTPORT) + 
      (theMaterialType & MURPORT) +
      (theMaterialType & INPUTMURPORT) +
      (theMaterialType & PECPORT);
  }

  return result;
}
//*/
