#include <GridFaceData.cuh>
#include <GridEdgeData.hxx>
#include <GridFace.hxx>
#include <GridEdge.hxx>

#include <GridGeometry.hxx>


#include <PhysConsts.hxx>


// From Surroundint GridEdgeDatas
void
GridFaceData::
DeduceShapeIndices()
{
  m_ShapeIndices.clear();
  const vector<T_Element>& theTEdges = this->GetOutLineTEdge();

  Standard_Size nb = theTEdges.size();
  for(Standard_Size index=0; index<nb; index++){
    GridEdgeData* currEdgeData = (GridEdgeData*) theTEdges[index].GetData();
    const set<Standard_Integer>& currShapeIndices = currEdgeData->GetShapeIndices();
    for(set<Standard_Integer>::const_iterator currIter=currShapeIndices.begin(); currIter!=currShapeIndices.end(); currIter++){
      m_ShapeIndices.insert(*currIter);
    }
  }
}



// From ShapeIndices
void
GridFaceData::
DeduceMaterialType()
{
  Standard_Integer tmp_mt=0;

  if(m_ShapeIndices.empty()){
    tmp_mt = GetBaseGridFace()->GetGridGeom()->GetBackGroundMaterialType();
  }else{
    for(set<Standard_Integer>::iterator iter=m_ShapeIndices.begin(); iter!=m_ShapeIndices.end(); iter++){
      Standard_Integer curr_ShapeIndex = *iter;
      Standard_Integer curr_mt =  GetBaseGridFace()->GetGridGeom()->GetMaterialTypeWithShapeIndex( curr_ShapeIndex );
      tmp_mt = tmp_mt | curr_mt;
    }
  }

  this->SetMaterialType(tmp_mt);
}


// From ShapeIndices
void
GridFaceData::
DeduceState()
{
  if(m_ShapeIndices.empty()){
    this->SetState(OUTSHAPE);
  }else{
    this->SetState(INSHAPE);
  }
}


void
GridFaceData::
DeduceType()
{
  const vector<T_Element>& theTEdges = this->GetOutLineTEdge();

  Standard_Size nb = theTEdges.size();
  for(Standard_Size index=0; index<nb; index++){
    GridEdgeData* currEdgeData = (GridEdgeData*) theTEdges[index].GetData();
    if(currEdgeData->IsPartial()){
      this->SetType(PFFACE);
      return;
    }
  }

  this->SetType(REGFACE);
}



void
GridFaceData::
DeduceMaterialData()
{
  const GridBndData* theGridBndDatas = GetBaseGridFace()->GetGridGeom()->GetGridBndDatas();

  vector<VertexData*> theAllVertexDatas;
  GetOrderedVertexDatas(theAllVertexDatas);

  Standard_Integer nb = theAllVertexDatas.size();
  Standard_Real theMu = 0.0;
  Standard_Real theEps = 0.0;
  Standard_Real theSigma = 0.0;

  vector<VertexData*>::iterator iter;
  for(iter=theAllVertexDatas.begin(); iter!=theAllVertexDatas.end();iter++){
    VertexData* currVertex = *iter;
    Standard_Real currMu = 1.0;  // for not specially defined material
    Standard_Real currEps = 1.0;
    Standard_Real currSigma = 0.0;
    if(currVertex->HasAnyUserDefinedMatData()){
      const set<Standard_Integer>& theMatDataIndices = currVertex->GetMatDataIndices();
      currMu = theGridBndDatas->GetMuAccordingMatIndices(theMatDataIndices, 2);
      currEps = theGridBndDatas->GetEpsAccordingMatIndices(theMatDataIndices, 2);
      currSigma = theGridBndDatas->GetSigmaAccordingMatIndices(theMatDataIndices, 2);
    }  
    theMu += currMu;
    theEps += currEps;
    theSigma += currSigma;
  }
  theMu=theMu/((Standard_Integer)nb);
  theEps=theEps/((Standard_Integer)nb);
  theSigma=theSigma/((Standard_Integer)nb);
  theMu = theMu * mksConsts.mu0;
  theEps = theEps * mksConsts.epsilon0;
  this->SetMu(theMu);
  this->SetEpsilon(theEps);
  this->SetSigma(theSigma);
}


/*
void
GridFaceData::
DeduceMaterialData()
{
  Standard_Real theEps = 0.0;
  Standard_Real theMu = 0.0;
  Standard_Real theSigma = 0.0;

  const vector<T_Element>& theAllTEdges = this->GetOutLineTEdge(); // m_EdgeElements
  vector<T_Element>::const_iterator iter;

  for(iter=theAllTEdges.begin(); iter!=theAllTEdges.end(); iter++){
    GridEdgeData* currEdgeData = (GridEdgeData*)(iter->GetData());
        theEps += currEdgeData->GetEpsilon();
        theMu += currEdgeData->GetMu();
        theSigma += currEdgeData->GetSigma();
	//cout<<"eps = "<<currEdgeData->GetEpsilon()/mksConsts.epsilon0<<endl;
	//cout<<"mu = "<<currEdgeData->GetMu()/mksConsts.mu0<<endl;
	//cout<<"sigma = "<<theSigma<<endl;
	
  }

  Standard_Integer nb = theAllTEdges.size();
  theEps=theEps/((Standard_Real)nb); 
  theMu=theMu/((Standard_Real)nb); 
  theSigma=theSigma/((Standard_Real)nb); 
  //cout<<"eps, mu, sigma = ["<<theEps<<" "<<theMu<<" "<<theSigma<<"]"<<endl;
  this->SetEpsilon(theEps);
  this->SetMu(theMu);
  this->SetSigma(theSigma);
}*/
