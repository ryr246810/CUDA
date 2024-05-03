#include <EdgeElecFldsBase_Cyl3D.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>

//#define CEMPIC2D_DBG
#ifdef CEMPIC2D_DBG
#define ELECFLDSBASE_DBG
#endif

EdgeElecFldsBase_Cyl3D::EdgeElecFldsBase_Cyl3D()
  :FieldsBase()
{
}


EdgeElecFldsBase_Cyl3D::EdgeElecFldsBase_Cyl3D(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule)
  :FieldsBase(theCntr, theRule)
{
}


EdgeElecFldsBase_Cyl3D::EdgeElecFldsBase_Cyl3D(const FieldsDefineCntr* theCntr)
  :FieldsBase(theCntr)
{
}


EdgeElecFldsBase_Cyl3D::~EdgeElecFldsBase_Cyl3D()
{
  m_Datas.clear();
}


void EdgeElecFldsBase_Cyl3D::Setup()
{
  FieldsBase::Setup(); // 无用，空实现

  m_Datas.clear();

  switch(m_Rule)
    {
    case EXCLUDING:
      {
	GetGridGeom_Cyl3D()->GetAllGridEdgeDatasNotOfMaterialTypesOfPhysRgn(m_Materials, true, m_Datas);

#ifdef ELECFLDSBASE_DBG
	cout<<"EdgeElecFldsBase_Cyl3D::Setup()------------------------------EXCLUDING"<<endl;
	vector<GridEdgeData*>::iterator iter;
	for(iter=m_Datas.begin(); iter!=m_Datas.end(); iter++){
	  GridEdgeData* currEdge = *iter;
	  Standard_Size indxVec[2];
	  currEdge->GetBaseGridEdge()->GetVecIndex(indxVec);

	  cout<<"dir = "<<currEdge->GetDir()<<"  indxVec = [" <<indxVec[0]<<", "<<indxVec[1]<<"] ";
	  cout<<"length, dualArea = ["<< currEdge->GetGeomDim()<<", "<<currEdge->GetDualGeomDim()<<" ] "<<endl;
	}
#endif
	break;
      }
    case INCLUDING:
      {
	GetGridGeom_Cyl3D()->GetAllGridEdgeDatasOfMaterialTypesOfPhysRgn(m_Materials, true, m_Datas);
#ifdef ELECFLDSBASE_DBG
	cout<<"EdgeElecFldsBase_Cyl3D::Setup()------------------------------including"<<endl;
	set<Standard_Integer>::iterator iter;
	for(iter = m_Materials.begin(); iter!= m_Materials.end(); iter++ ){
	  cout<<"*iter------------------------"<<*iter<<endl;
	  cout<<"m_Datas.size() = "<<m_Datas.size()<<endl;
	}
#endif

	break;
      }
    case NORULE:
      {
	GetGridGeom_Cyl3D()->GetAllGridEdgeDatasOfPhysRgn(true, m_Datas);
	break;
      }
    }
}


bool EdgeElecFldsBase_Cyl3D::IsPhysDataMemoryLocated() const
{
  bool result = true;

  Standard_Integer nb = m_Datas.size();
  for(Standard_Integer index = 0; index<nb; index++){
    bool tmp = m_Datas[index]->IsPhysDataDefined();
    result = result && tmp;
    tmp = m_Datas[index]->IsSharedFacesPhysDataDefined();
    result = result && tmp;
  }

  return result;
}
