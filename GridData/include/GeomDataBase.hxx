#ifndef _GeomDataBase_Headerfile
#define _GeomDataBase_Headerfile

#include <TxVector.h>

#include <Standard_TypeDefine.hxx>

#include <BaseDataDefine.hxx>

class GeomDataBase
{
public:
  GeomDataBase();
  GeomDataBase(Standard_Integer m_Mark);
  virtual ~GeomDataBase();


  virtual void SetupGeomDimInf();

public:
  void SetState(Standard_Integer _mark);
  Standard_Integer GetState() const;

  void SetType(Standard_Integer _type);
  Standard_Integer  GetType() const;

  void SetMark(Standard_Integer _mark);
  Standard_Integer  GetMark() const;
  void ResetMark();

  void AddMark(Standard_Integer _mark);

  void RemoveStateMark();
  void RemoveTypeMark();


public:

  virtual Standard_Real GetGeomDim() const;
  virtual Standard_Real GetDualGeomDim() const;

  virtual Standard_Real GetSweptGeomDim() const;
  virtual Standard_Real GetDualSweptGeomDim() const;

  //these is for the Cyl3D function
  virtual Standard_Real GetSweptGeomDim_Near();
  virtual Standard_Real GetDualGeomDim_Near();
private:
  //Standard_Integer m_ShapeIndex;
  Standard_Integer m_Mark;
};

#endif
