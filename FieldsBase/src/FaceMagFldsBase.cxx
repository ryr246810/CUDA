#include <FaceMagFldsBase.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>


FaceMagFldsBase::
FaceMagFldsBase()
  :FieldsBase()
{
}


FaceMagFldsBase::
FaceMagFldsBase(const FieldsDefineCntr* theCntr)
  :FieldsBase(theCntr)
{
}


FaceMagFldsBase::
FaceMagFldsBase(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule)
  :FieldsBase(theCntr, theRule)
{
}


FaceMagFldsBase::
~FaceMagFldsBase()
{
  m_Datas.clear();
}


void 
FaceMagFldsBase::
Setup()
{
  FieldsBase::Setup();
  m_Datas.clear();

  switch(m_Rule)
    {
    case EXCLUDING:
      {
	GetGridGeom()->GetAllGridFaceDatasNotOfMaterialTypesOfPhysRgn(m_Materials, m_Datas);
	break;
      }
    case INCLUDING:
      {
	GetGridGeom()->GetAllGridFaceDatasOfMaterialTypesOfPhysRgn(m_Materials, m_Datas);
	break;
      }
    case NORULE:
      {
	GetGridGeom()->GetAllGridFaceDatasOfPhysRgn(m_Datas);
	break;
      }
    }
}


bool 
FaceMagFldsBase::
IsPhysDataMemoryLocated() const
{
  bool result = true;

  Standard_Integer nb = m_Datas.size();
  for(Standard_Integer index = 0; index<nb; index++){
    bool tmp = m_Datas[index]->IsPhysDataDefined();
    result = result && tmp;
    tmp = m_Datas[index]->IsOutLineEdgePhysDataDefined();
    result = result && tmp;
  }

  return result;
}

