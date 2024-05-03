#ifndef _FieldsDefineCntr_HeaderFile
#define _FieldsDefineCntr_HeaderFile

#include <GridGeometry.hxx>
#include <GridGeometry_Cyl3D.hxx>
#include <FieldsDefineRules.hxx>

#include <vector>


class FieldsDefineCntr
{
public:
  FieldsDefineCntr(const GridGeometry* _gridGeom, 
		   const FieldsDefineRules* defrules);

  virtual ~FieldsDefineCntr();


public:
  virtual void LocateMemeory_For_FieldsPhysDatas(){};


public:
  const FieldsDefineRules* GetFieldsDefineRules() const
  {
    return m_FieldsDataDefRules;
  }

  const GridGeometry* GetGridGeom() const
  {
    return m_GridGeom;
  };

  const GridGeometry* GetGridGeom(Standard_Integer i) const {
    if(i == -1) return m_GridGeom;
    else return m_GridGeom_Cyl3D->GetGridGeometry(i);
  };

  const GridGeometry_Cyl3D* GetGridGeom_Cyl3D()const
  {
    return m_GridGeom_Cyl3D;
  };


  const ZRGrid* GetZRGrid() const
  {
    return m_GridGeom->GetZRGrid();
  };

  const GridBndData* GetGridBndDatas() const
  {
    return m_GridGeom->GetGridBndDatas();
  };


protected:
  const GridGeometry* m_GridGeom;
  const GridGeometry_Cyl3D* m_GridGeom_Cyl3D;
  const FieldsDefineRules* m_FieldsDataDefRules;


  // prevent using the default constructor
private:
  FieldsDefineCntr();
  void SetGridGeom(const GridGeometry* _gridGeom) {m_GridGeom = _gridGeom;};

public:
   void SetGridGeom_Cyl3D(const GridGeometry_Cyl3D* _gridGeom3D) {m_GridGeom_Cyl3D = _gridGeom3D;};

};

#endif
