#include <SweptFaceMagFldsBase_Cyl3D.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>

#ifdef CEMPIC2D_DBG
#define ELECFLDSBASE_DBG
#endif


SweptFaceMagFldsBase_Cyl3D::
SweptFaceMagFldsBase_Cyl3D()
  :FieldsBase()
{
}


SweptFaceMagFldsBase_Cyl3D::
SweptFaceMagFldsBase_Cyl3D(const FieldsDefineCntr* theCntr, 
		     PhysDataDefineRule theRule)
  :FieldsBase(theCntr, theRule)
{
}


SweptFaceMagFldsBase_Cyl3D::
SweptFaceMagFldsBase_Cyl3D(const FieldsDefineCntr* theCntr)
  :FieldsBase(theCntr)
{
}


SweptFaceMagFldsBase_Cyl3D::
~SweptFaceMagFldsBase_Cyl3D()
{
  m_Datas.clear();
}


void 
SweptFaceMagFldsBase_Cyl3D::
Setup()
{
  FieldsBase::Setup();

  m_Datas.clear();

  switch(m_Rule)
    {
    case EXCLUDING:
      {
	GetGridGeom_Cyl3D()->GetAllGridEdgeDatasNotOfMaterialTypesOfPhysRgn(m_Materials, true, m_Datas);
	break;
      }
    case INCLUDING:
      {
	GetGridGeom_Cyl3D()->GetAllGridEdgeDatasOfMaterialTypesOfPhysRgn(m_Materials, true, m_Datas);
	break;
      }
    case NORULE:
      {
	GetGridGeom_Cyl3D()->GetAllGridEdgeDatasOfPhysRgn(true, m_Datas);
	break;
      }
    }
}


bool 
SweptFaceMagFldsBase_Cyl3D::
IsPhysDataMemoryLocated() const
{
  bool result = true;

  Standard_Integer nb = m_Datas.size();
  for(Standard_Integer index = 0; index<nb; index++){
    bool tmp = m_Datas[index]->IsPhysDataDefined();
    result = result && tmp;
    tmp = m_Datas[index]->IsOutLineDEdgePhysDataDefined();
    result = result && tmp;
  }

  return result;
}
