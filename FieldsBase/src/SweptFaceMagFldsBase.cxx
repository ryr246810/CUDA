#include <SweptFaceMagFldsBase.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>

#ifdef CEMPIC2D_DBG
#define ELECFLDSBASE_DBG
#endif


SweptFaceMagFldsBase::
SweptFaceMagFldsBase()
  :FieldsBase()
{
}


SweptFaceMagFldsBase::
SweptFaceMagFldsBase(const FieldsDefineCntr* theCntr, 
		     PhysDataDefineRule theRule)
  :FieldsBase(theCntr, theRule)
{
}


SweptFaceMagFldsBase::
SweptFaceMagFldsBase(const FieldsDefineCntr* theCntr)
  :FieldsBase(theCntr)
{
}


SweptFaceMagFldsBase::
~SweptFaceMagFldsBase()
{
  m_Datas.clear();
}


void 
SweptFaceMagFldsBase::
Setup()
{
  FieldsBase::Setup();

  m_Datas.clear();

  switch(m_Rule)
    {
    case EXCLUDING:
      {
	GetGridGeom()->GetAllGridEdgeDatasNotOfMaterialTypesOfPhysRgn(m_Materials, true, m_Datas);
	break;
      }
    case INCLUDING:
      {
	GetGridGeom()->GetAllGridEdgeDatasOfMaterialTypesOfPhysRgn(m_Materials, true, m_Datas);
	break;
      }
    case NORULE:
      {
	GetGridGeom()->GetAllGridEdgeDatasOfPhysRgn(true, m_Datas);
	break;
      }
    }
}


bool 
SweptFaceMagFldsBase::
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
