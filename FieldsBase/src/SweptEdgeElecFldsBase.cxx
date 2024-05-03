#include <SweptEdgeElecFldsBase.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>


SweptEdgeElecFldsBase::
SweptEdgeElecFldsBase()
  :FieldsBase()
{
}


SweptEdgeElecFldsBase::
SweptEdgeElecFldsBase(const FieldsDefineCntr* theCntr, 
		      PhysDataDefineRule theRule)
  :FieldsBase(theCntr, theRule)
{
}


SweptEdgeElecFldsBase::
SweptEdgeElecFldsBase(const FieldsDefineCntr* theCntr)
  :FieldsBase(theCntr)
{
}


SweptEdgeElecFldsBase::
~SweptEdgeElecFldsBase()
{
  m_Datas.clear();
}


void 
SweptEdgeElecFldsBase::
Setup()
{
  FieldsBase::Setup();

  m_Datas.clear();

  switch(m_Rule)
    {
    case EXCLUDING:
      {
	GetGridGeom()->GetAllGridVertexDatasNotOfMaterialTypesOfPhysRgn(m_Materials, true, m_Datas);
	break;
      }
    case INCLUDING:
      {
	GetGridGeom()->GetAllGridVertexDatasOfMaterialTypesOfPhysRgn(m_Materials, true, m_Datas);
	break;
      }
    case NORULE:
      {
	GetGridGeom()->GetAllGridVertexDatasOfPhysRgn(true, m_Datas);
	break;
      }
    }
}


bool 
SweptEdgeElecFldsBase::
IsPhysDataMemoryLocated() const
{
  bool result = true;

  Standard_Integer nb = m_Datas.size();
  for(Standard_Integer index = 0; index<nb; index++){
    bool tmp = m_Datas[index]->IsPhysDataDefined();
    result = result && tmp;
    tmp = m_Datas[index]->IsSharedDFacesPhysDataDefined();
    result = result && tmp;
  }

  return result;
}
