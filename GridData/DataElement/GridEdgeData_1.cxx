
#include <GridEdge.hxx>
#include <T_Element.hxx>

#include <GridEdgeData.hxx>
#include <GridVertexData.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>

#include <GridGeometry.hxx>

#include <PhysConsts.hxx>


void
GridEdgeData::
DeduceMaterialData()
{
  Standard_Real theEps = 0.0;
  Standard_Real theMu = 0.0;
  Standard_Real theSigma = 0.0;

  vector<VertexData*> m_MidVertices;
  VertexData* firstVertex;
  VertexData* lastVertex;

  if(m_MidVertices.empty()){
    firstVertex = this->GetFirstVertex();
    lastVertex = this->GetLastVertex();
    LoadMaterialDataOfSegment(firstVertex, lastVertex, theEps, theMu, theSigma);
  }else{
    firstVertex = this->GetFirstVertex();
    lastVertex = m_MidVertices[0];
    LoadMaterialDataOfSegment(firstVertex, lastVertex, theEps, theMu, theSigma);

    Standard_Integer nb = m_MidVertices.size();
    for(Standard_Integer i=0; i<nb-1; i++){
      firstVertex = m_MidVertices[i];
      lastVertex = m_MidVertices[i+1];
      LoadMaterialDataOfSegment(firstVertex, lastVertex, theEps, theMu, theSigma);
    }

    firstVertex = m_MidVertices[nb-1];
    lastVertex = this->GetLastVertex();
    LoadMaterialDataOfSegment(firstVertex, lastVertex, theEps, theMu, theSigma);
  }

  theEps = theEps * mksConsts.epsilon0;
  theMu = theMu * mksConsts.mu0;

  this->SetEpsilon(theEps);
  this->SetMu(theMu);
  this->SetSigma(theSigma);
}




void DeduceMaterialDataOfVertex(const GridBndData* theGridBndDatas, const VertexData* theVertex, 
				Standard_Real& theEps, Standard_Real& theMu, 
				Standard_Real& theSigma, Standard_Integer& nb, Standard_Integer dir)
{
  if(!theVertex->IsMaterialType(PEC)){
    nb=1;
    if(theVertex->HasAnyUserDefinedMatData()){
      // user defined
      const set<Standard_Integer>& theMatDataIndices = theVertex->GetMatDataIndices();
      theEps = theGridBndDatas->GetEpsAccordingMatIndices(theMatDataIndices, dir);
      theMu = theGridBndDatas->GetMuAccordingMatIndices(theMatDataIndices, dir);
      theSigma = theGridBndDatas->GetSigmaAccordingMatIndices(theMatDataIndices, dir);
    }else{
      // free space 
      theEps = 1.0; 
      theMu = 1.0; 
      theSigma = 0.0;
    }
  }else{
    // PEC
    nb=0;
    theEps = 0.0;
    theMu = 0.0;
    theSigma = 0.0;
  }
}



void
GridEdgeData::
LoadMaterialDataOfSegment(VertexData* firstVertex, 
			  VertexData* lastVertex, 
			  Standard_Real& theEps, 
			  Standard_Real& theMu, 
			  Standard_Real& theSigma)
{
  const GridBndData* theGridBndDatas = GetBaseGridEdge()->GetGridGeom()->GetGridBndDatas();

  Standard_Integer nb = 0;

  Standard_Integer dir = this->GetDir();

  Standard_Real firstEps = 0.0;
  Standard_Real firstMu = 0.0;
  Standard_Real firstSigma = 0.0;

  Standard_Real lastEps = 0.0;
  Standard_Real lastMu = 0.0;
  Standard_Real lastSigma = 0.0;

  Standard_Real currEps = 0.0;
  Standard_Real currMu = 0.0;
  Standard_Real currSigma = 0.0;

  Standard_Integer firstNb=0;
  Standard_Integer lastNb=0;

  DeduceMaterialDataOfVertex(theGridBndDatas, firstVertex, firstEps, firstMu, firstSigma, firstNb, dir);
  DeduceMaterialDataOfVertex(theGridBndDatas, lastVertex, lastEps, lastMu, lastSigma, lastNb, dir);
  nb = firstNb + lastNb;

  /*
  if(nb==0){
    cout<<"GridEdgeData::LoadMaterialDataOfSegment------------------------------------error-----------------------"<<endl;
  }
  //*/

  currEps = (firstEps + lastEps)/((Standard_Real)nb);
  currMu = (firstMu + lastMu)/((Standard_Real)nb);
  currSigma = (firstSigma + lastSigma)/((Standard_Real)nb) ;


  Standard_Real theRealLength = EdgeData::GetGeomDim();

  Standard_Real currRatio  =  ((lastVertex->GetLocation()-firstVertex->GetLocation()).length())/theRealLength;
  theEps += currEps*currRatio;
  theMu += currMu*currRatio;
  theSigma += currSigma*currRatio;
}

