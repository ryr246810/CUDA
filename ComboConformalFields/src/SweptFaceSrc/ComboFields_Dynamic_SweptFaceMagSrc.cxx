#include <ComboFields_Dynamic_SweptFaceMagSrc.hxx>


ComboFields_Dynamic_SweptFaceMagSrc::ComboFields_Dynamic_SweptFaceMagSrc()
  :ComboFields_Dynamic_SrcBase()
{
}


ComboFields_Dynamic_SweptFaceMagSrc::ComboFields_Dynamic_SweptFaceMagSrc(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule)
  :ComboFields_Dynamic_SrcBase(theCntr, theRule)
{
}


ComboFields_Dynamic_SweptFaceMagSrc::~ComboFields_Dynamic_SweptFaceMagSrc()
{
}


void ComboFields_Dynamic_SweptFaceMagSrc::SetupRgn()
{
  TxSlab2D<Standard_Integer> theRgn;

  for(Standard_Integer dir=0;dir<2;dir++){
    theRgn.setLowerBound(dir, m_FirstVertexIndex[dir]);
    if(dir==m_Dir){
      theRgn.setUpperBound(dir, m_FirstVertexIndex[dir] + m_GridEdgeNum);
    }else{
      theRgn.setUpperBound(dir, m_FirstVertexIndex[dir]);
    }
  }

  SetRgn(theRgn);
}



// should do computing in current GeomGeometry Region define
// this function should be modified to be suitable for MPI
void ComboFields_Dynamic_SweptFaceMagSrc::Setup()
{
  m_Datas.clear();

  TxSlab2D<Standard_Integer> theRgn  = GetFldsDefCntr()->GetZRGrid()->GetXtndRgn() & this->GetRgn();

  switch(m_Rule)
    {
    case EXCLUDING:
      {
	GetGridGeom()->GetGridEdgeDatasNotOfMaterialTypesOfSubRgn(m_Dir, m_Materials, theRgn, false, m_Datas);
	break;
      }
    case INCLUDING:
      {
	GetGridGeom()->GetGridEdgeDatasOfMaterialTypesOfSubRgn(m_Dir, m_Materials, theRgn, false, m_Datas);
	break;
      }
   case NORULE:
      {
	set<Standard_Integer> tmpMaterials;
	tmpMaterials.clear();
	tmpMaterials.insert(0);
	GetGridGeom()->GetGridEdgeDatasNotOfMaterialTypesOfSubRgn(m_Dir, tmpMaterials, theRgn, false, m_Datas);
	break;
      }
    }

  /*
  cout<<"m_Datas.size()\t=\t"<<m_Datas.size()<<endl;
  if(m_Datas[0]->IsMaterialType(PEC)){
    cout<<"Is    PEC-----------------------------------1"<<endl;
  }else if(m_Datas[0]->IsMaterialType(EMFREESPACE)){
    cout<<"Is    FreeSpace-----------------------------------1"<<endl;
  }
  //*/
}



void ComboFields_Dynamic_SweptFaceMagSrc::Advance()
{

}
