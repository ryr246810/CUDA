#include <FaceMagFldsBase_Cyl3D.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>


FaceMagFldsBase_Cyl3D::
FaceMagFldsBase_Cyl3D()
  :FieldsBase()
{
}


FaceMagFldsBase_Cyl3D::
FaceMagFldsBase_Cyl3D(const FieldsDefineCntr* theCntr)
  :FieldsBase(theCntr)
{
}


FaceMagFldsBase_Cyl3D::
FaceMagFldsBase_Cyl3D(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule)
  :FieldsBase(theCntr, theRule)
{
}


FaceMagFldsBase_Cyl3D::
~FaceMagFldsBase_Cyl3D()
{
  m_Datas.clear();
}


void 
FaceMagFldsBase_Cyl3D::
Setup()
{
  FieldsBase::Setup();
  m_Datas.clear();

  switch(m_Rule)
    {
      case EXCLUDING:
      {
        GetGridGeom_Cyl3D()->GetAllGridFaceDatasNotOfMaterialTypesOfPhysRgn(m_Materials, m_Datas);
        break;
      }
      case INCLUDING:
      {
        GetGridGeom_Cyl3D()->GetAllGridFaceDatasOfMaterialTypesOfPhysRgn(m_Materials, m_Datas);
        break;
      }
      case NORULE:
      {
        GetGridGeom_Cyl3D()->GetAllGridFaceDatasOfPhysRgn(m_Datas);
        break;
      }
    }
}


bool 
FaceMagFldsBase_Cyl3D::
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

