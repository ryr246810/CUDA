#include <GridGeometry.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>

void GridGeometry::Setup()
{
  InitDefineGridVertices();
  InitDefineGridEdges();
  InitDefineGridFaces();

  BuildGridVertices();
  BuildGridEdgeAppendingVertices();
  BuildGridFaceAppendingVertices();
  SetupGridVertices();

  BuildGridEdges();
  BuildGridFaceDatas();

  SetPMLAccordingPorts(); 
  SetNonPMLPortBnd();

  BuildSurroundingGeomElements();
  //BuildWeightsTool();
}


void GridGeometry::Build_Near_Edge()
{

  Standard_Integer relativeDir = 1;
  for(Standard_Integer dim =0; dim<2; dim++){
    Standard_Size nbge = m_ZRGrid->GetEdgeSize(dim);
    for(Standard_Size index=0; index<nbge; index++){
      vector<GridEdgeData*> currGridEdgeDatas = m_Edges[dim][index].GetEdges();
      vector<GridEdgeData*> plusGridEdgeDatas = (plus_Geometry->GetGridEdges()[dim] + index)->GetEdges();
      vector<GridEdgeData*> minuGridEdgeDatas = (minu_Geometry->GetGridEdges()[dim] + index)->GetEdges();
      if(currGridEdgeDatas.size()>1 ||plusGridEdgeDatas.size()>1 ||minuGridEdgeDatas.size()>1){
		cout<<"the Error in the void GridGeometry::Build_Near_Edge"<<endl;
	  }

		/*if(dim==0){
			for(Standard_Size i=0;i<currGridEdgeDatas.size();i++){
			currGridEdgeDatas[i]->AddNearEEdge(currGridEdgeDatas[i],-1*relativeDir);
			currGridEdgeDatas[i]->AddNearEEdge(minuGridEdgeDatas[i],relativeDir);
			
			currGridEdgeDatas[i]->AddNearMEdge(currGridEdgeDatas[i],-1*relativeDir);
			currGridEdgeDatas[i]->AddNearMEdge(plusGridEdgeDatas[i],relativeDir);
			}
		}
		else{

			for(Standard_Size i=0;i<currGridEdgeDatas.size();i++){
			currGridEdgeDatas[i]->AddNearEEdge(currGridEdgeDatas[i],relativeDir);
			currGridEdgeDatas[i]->AddNearEEdge(minuGridEdgeDatas[i],-1*relativeDir);

			currGridEdgeDatas[i]->AddNearMEdge(currGridEdgeDatas[i],relativeDir);
			currGridEdgeDatas[i]->AddNearMEdge(plusGridEdgeDatas[i],-1*relativeDir);
			}
		}*/
		
		if(dim==0)
		{
			for(Standard_Size i=0;i<currGridEdgeDatas.size();i++)
			{
				currGridEdgeDatas[i]->AddNearEEdge(currGridEdgeDatas[i],-1*relativeDir);
				currGridEdgeDatas[i]->AddNearEEdge(plusGridEdgeDatas[i],relativeDir);
				
				currGridEdgeDatas[i]->AddNearMEdge(currGridEdgeDatas[i],-1*relativeDir);
				currGridEdgeDatas[i]->AddNearMEdge(minuGridEdgeDatas[i],relativeDir);
			}
		}
		else{
			for(Standard_Size i=0;i<currGridEdgeDatas.size();i++)
			{
				currGridEdgeDatas[i]->AddNearEEdge(currGridEdgeDatas[i],relativeDir);
				currGridEdgeDatas[i]->AddNearEEdge(plusGridEdgeDatas[i],-1*relativeDir);
				
				currGridEdgeDatas[i]->AddNearMEdge(currGridEdgeDatas[i],relativeDir);
				currGridEdgeDatas[i]->AddNearMEdge(minuGridEdgeDatas[i],-1*relativeDir);
			}
		}
    }
  }
}
