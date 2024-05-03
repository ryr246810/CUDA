#include <GridVertexData.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>
#include <GridGeometry.hxx>

#include <PhysConsts.hxx>

GridVertexData::
GridVertexData()
  :VertexData()
{
  m_GridGeom = NULL;
  m_Index = 0;
  m_C[0] = 0.0;
  m_C[1] = 0.0;
}


GridVertexData::
GridVertexData(GridGeometry* _gridgeom,
	       Standard_Size _index)
  :VertexData()
{
  m_GridGeom = _gridgeom;
  m_Index = _index;
}


GridVertexData::
GridVertexData(GridGeometry* _gridgeom, 
	       Standard_Size _index,
	       Standard_Integer _mark)
  :VertexData(_mark)
{
  m_GridGeom = _gridgeom;
  m_Index = _index;
}


GridVertexData::
~GridVertexData()
{
  ClearSharingElems();
}


void 
GridVertexData::
ClearSharingElems()
{
  ClearSharingDivTEdges();
  ClearSharingTDFaces();
  ClearSharingGridFaceDatas();
}


Standard_Size 
GridVertexData::
GetIndex() const
{ 
  return m_Index; 
}


const ZRGrid* 
GridVertexData::
GetZRGrid() const
{
  return m_GridGeom->GetZRGrid();
}


TxVector2D<Standard_Real> 
GridVertexData::
GetLocation() const
{
  return GetZRGrid()->GetCoord_From_VertexScalarIndx(m_Index);
}


void 
GridVertexData::
GetVecIndex(Standard_Size indxVec[2]) const
{
  this->GetZRGrid()->FillVertexIndxVec(m_Index, indxVec);
}


bool GridVertexData::IsSharedDFacesPhysDataDefined()
{
  bool result=true;

  Standard_Size nb = m_SharedTDFaces.size();
  for(Standard_Size index = 0; index<nb; index++){
    bool tmp = (m_SharedTDFaces[index].GetData())->IsSweptPhysDataDefined();
    result = result && tmp;
  }
  
  return result;
}



/***************************************************************/
/***************************************************************/
/***************************************************************/
/***************************************************************/
/***************************************************************/
void 
GridVertexData::
Setup()
{
  SetupGeomDimInf();
  SetupMaterialData();
}


void 
GridVertexData::
SetupGeomDimInf()
{
  ComputeSweptGeomDim();
  ComputeDualSweptGeomDim();
}


void 
GridVertexData::
ComputeDualSweptGeomDim()
{
  Standard_Size indxVec[2];
  this->GetVecIndex(indxVec);

  Standard_Real dZ = GetZRGrid()->GetDualStep(0, indxVec[0]);
  Standard_Real dR = GetZRGrid()->GetDualStep(1, indxVec[1]);

  m_DualAreaOfSweptEdge = dZ*dR;
}

Standard_Real
GridVertexData::
GetSweptGeomDim() const{

    Standard_Real n_Segment = GetGridGeom()->GetPhiNumber();
    Standard_Real result=VertexData::GetSweptGeomDim()/n_Segment;
    return result;

  };

void 
GridVertexData::
SetupMaterialData()
{
  const GridBndData* theGridBndDatas = this->GetGridGeom()->GetGridBndDatas();

  Standard_Integer dir = 2; //represents Phi direction;

  if(!this->IsMaterialType(PEC)){
    DataBase::SetupMaterialData();

    Standard_Real theEps = 0.0;
    Standard_Real theMu = 0.0;
    Standard_Real theSigma = 0.0;

    if(this->HasAnyUserDefinedMatData()){
      // user defined
      const set<Standard_Integer>& theMatDataIndices = this->GetMatDataIndices();
      theEps = GetGridGeom()->GetGridBndDatas()->GetEpsAccordingMatIndices(theMatDataIndices, 2);
      theMu = GetGridGeom()->GetGridBndDatas()->GetMuAccordingMatIndices(theMatDataIndices, 2);
      theSigma = GetGridGeom()->GetGridBndDatas()->GetSigmaAccordingMatIndices(theMatDataIndices, 2);
      //cout<<GetGridGeom()->GetGridBndDatas()->GetDir()<<endl;
      }else{
      // free space 
      theEps = 1.0; 
      theMu = 1.0; 
      theSigma = 0.0;
    }
    //cout<<theMu<<endl;
    theEps = theEps * mksConsts.epsilon0;
    theMu = theMu * mksConsts.mu0;
    this->SetEpsilon(theEps);
    this->SetMu(theMu);
    this->SetSigma(theSigma);
  }

  /*
  if(!this->IsMaterialType(PEC)){
    Standard_Size indxVec[2];
    this->GetVecIndex(indxVec);
    cout<<"indxVec = ["<<indxVec[0]<<", "<<indxVec[1]<<"];";
    cout<<"  LengthOfSweptEdge = "<<m_LengthOfSweptEdge<<";";
    cout<<"  DualAreaOfSweptEdge = "<<m_DualAreaOfSweptEdge<<endl;
    cout<<"          [eps, mu, sigma] = ["<<this->GetEpsilon()<<", "<<this->GetMu()<<", "<<this->GetSigma()<<"]";
    cout<<endl;
  }
  //*/
}
