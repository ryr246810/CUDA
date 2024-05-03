#include <ComboFields_Dynamic_VertexElecSrc.hxx>
#include <GridVertexData.hxx>
//#include <EMSourceEquation.hxx>


ComboFields_Dynamic_VertexElecSrc::ComboFields_Dynamic_VertexElecSrc()
  :ComboFields_Dynamic_SrcBase()
{
}


ComboFields_Dynamic_VertexElecSrc::ComboFields_Dynamic_VertexElecSrc(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule)
  :ComboFields_Dynamic_SrcBase(theCntr, theRule)
{
}


ComboFields_Dynamic_VertexElecSrc::~ComboFields_Dynamic_VertexElecSrc()
{
}


void ComboFields_Dynamic_VertexElecSrc::SetupRgn()
{
  TxSlab2D<Standard_Integer> theRgn;

  for(Standard_Integer dir=0;dir<2;dir++){
    theRgn.setLowerBound(dir, m_VertexIndex[dir]);
    theRgn.setUpperBound(dir, m_VertexIndex[dir]);
  }

  SetRgn(theRgn);
}



// should do computing in current GeomGeometry Region define
// this function should be modified to be suitable for MPI
void ComboFields_Dynamic_VertexElecSrc::Setup()
{
  m_Datas.clear();
  TxSlab2D<Standard_Integer> theRgn  = GetFldsDefCntr()->GetZRGrid()->GetXtndRgn() & this->GetRgn();
  cout<<m_phiIndex<<endl;
  switch(m_Rule)
    {
    case EXCLUDING:
      {
	GetGridGeom(m_phiIndex)->GetGridVertexDatasNotOfMaterialTypesOfSubRgn(m_Materials, theRgn, true, m_Datas);
	break;
      }
    case INCLUDING:
      {
	GetGridGeom(m_phiIndex)->GetGridVertexDatasOfMaterialTypesOfSubRgn(m_Materials, theRgn, true, m_Datas);
	break;
      }
   case NORULE:
      {
	set<Standard_Integer> tmpMaterials;
	tmpMaterials.clear();
	tmpMaterials.insert(0);
	GetGridGeom(m_phiIndex)->GetGridVertexDatasNotOfMaterialTypesOfSubRgn(tmpMaterials, theRgn, true, m_Datas);
	break;
      }
    }
}



void ComboFields_Dynamic_VertexElecSrc::Advance()
{

}
