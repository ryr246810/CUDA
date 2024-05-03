#include <ComboFields_DefineCntr.hxx>


void 
ComboFields_DefineCntr::
LocateMemeory_For_FieldsPhysDatas()
{
  LocateMemeory_For_EdgePhysDatas();
  LocateMemeory_For_FacePhysDatas();
  LocateMemeory_For_VertexPhysDatas();
}

void 
ComboFields_DefineCntr::
LocateMemeory_For_3DFieldsPhysDatas()
{
  LocateMemeory_For_3DEdgePhysDatas();
  LocateMemeory_For_3DFacePhysDatas();
  LocateMemeory_For_3DVertexPhysDatas();
}

void 
ComboFields_DefineCntr::
LocateMemeory_For_VertexPhysDatas()
{
  Standard_Size theGlobalVIndxVec[2];
  Standard_Size theGlobalVIndx;
  Standard_Size theLocalFIndx;

  TxSlab2D<Standard_Integer> theRgn = GetGridGeom()->GetZRGrid()->GetXtndRgn();

  Standard_Integer Dir0 = 0;
  Standard_Integer Dir1 = 1;
  
  for(Standard_Size index0 = theRgn.getLowerBound(Dir0); index0<=theRgn.getUpperBound(Dir0); index0++){
    theGlobalVIndxVec[Dir0] = index0;
    for(Standard_Size index1 = theRgn.getLowerBound(Dir1); index1<=theRgn.getUpperBound(Dir1); index1++){
      theGlobalVIndxVec[Dir1] = index1;
      
      GetGridGeom()->GetZRGrid()->FillVertexIndx(theGlobalVIndxVec, theGlobalVIndx);
      GridVertexData* tmpGridVertexData = GetGridGeom()->GetGridVertices()+theGlobalVIndx;

      Standard_Integer tmpMaterial = tmpGridVertexData->GetEMMaterialType();
      if( ! (GetFieldsDefineRules()->IsBndElecMaterial(tmpMaterial)) ){
	Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetCntrElecPhysDataNum();
	tmpGridVertexData->SetupSweptPhysData(tmpPhysDataNum);
      }else{
	Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetBndElecPhysDataNum(tmpMaterial);
	if( tmpPhysDataNum!=0 ){
	  tmpGridVertexData->SetupSweptPhysData(tmpPhysDataNum );
	}
      }
    } // index1
  } // index0
}


void 
ComboFields_DefineCntr::
LocateMemeory_For_FacePhysDatas()
{
  Standard_Size theGlobalFIndxVec[2];
  Standard_Size theGlobalFIndx;

  TxSlab2D<Standard_Integer> theRgn = GetGridGeom()->GetZRGrid()->GetXtndRgn();

  Standard_Integer Dir0 = 0;
  Standard_Integer Dir1 = 1;
  
  for(Standard_Size index0 = theRgn.getLowerBound(Dir0); index0<theRgn.getUpperBound(Dir0); index0++){
    theGlobalFIndxVec[Dir0] = index0;
    for(Standard_Size index1 = theRgn.getLowerBound(Dir1); index1<theRgn.getUpperBound(Dir1); index1++){
      theGlobalFIndxVec[Dir1] = index1;
      
      GetGridGeom()->GetZRGrid()->FillFaceIndx(theGlobalFIndxVec, theGlobalFIndx);
      GridFace* tmpGridFace = GetGridGeom()->GetGridFaces()+theGlobalFIndx;
      const vector<GridFaceData*>& tmpDatas = tmpGridFace->GetFaces();

      if(tmpDatas.empty()) continue;

      Standard_Size nb = tmpDatas.size();
      for(Standard_Size j=0;j<nb;j++){
	Standard_Integer tmpMaterial = tmpDatas[j]->GetEMMaterialType();
	if( ! (GetFieldsDefineRules()->IsBndMagMaterial(tmpMaterial)) ){
	  Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetCntrMagPhysDataNum();
	  tmpDatas[j]->SetupPhysData(tmpPhysDataNum);
	}else{
	  Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetBndMagPhysDataNum(tmpMaterial);
	  if( tmpPhysDataNum!=0 ){
	    tmpDatas[j]->SetupPhysData( tmpPhysDataNum );
	  }
	}
      }
    } // index1
  } // index0
}


void 
ComboFields_DefineCntr::
LocateMemeory_For_EdgePhysDatas()
{
  Standard_Size theGlobalEIndxVec[2];
  Standard_Size theGlobalEIndx;

  TxSlab2D<Standard_Integer> theRgn = GetGridGeom()->GetZRGrid()->GetXtndRgn();

  for(Standard_Integer dir=0; dir<2; dir++){
    Standard_Integer Dir0 = dir;
    Standard_Integer Dir1 = (Dir0+1)%2;

    for(Standard_Size index0 = theRgn.getLowerBound(Dir0); index0<theRgn.getUpperBound(Dir0); index0++){
      theGlobalEIndxVec[Dir0] = index0;
      for(Standard_Size index1 = theRgn.getLowerBound(Dir1); index1<=theRgn.getUpperBound(Dir1); index1++){
	theGlobalEIndxVec[Dir1] = index1;
	  
	GetGridGeom()->GetZRGrid()->FillEdgeIndx(Dir0, theGlobalEIndxVec, theGlobalEIndx);
	GridEdge* tmpGridEdge = GetGridGeom()->GetGridEdges()[Dir0]+theGlobalEIndx;
	
	const vector<GridEdgeData*>& tmpDatas = tmpGridEdge->GetEdges();
	Standard_Size nb = tmpDatas.size();

	for(Standard_Size j=0;j<nb;j++){
	  Standard_Integer tmpMaterial = tmpDatas[j]->GetEMMaterialType();
	  if( ! (GetFieldsDefineRules()->IsBndElecMaterial(tmpMaterial)) ){
	    Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetCntrElecPhysDataNum();
	    if( tmpPhysDataNum!=0 ){
	      tmpDatas[j]->SetupPhysData(tmpPhysDataNum);
	    }
	  }else{
	    Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetBndElecPhysDataNum(tmpMaterial);
	    if( tmpPhysDataNum!=0 ){
	      tmpDatas[j]->SetupPhysData( tmpPhysDataNum );
	    }
	  }

	  if( ! (GetFieldsDefineRules()->IsBndMagMaterial(tmpMaterial)) ){
	    Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetCntrMagPhysDataNum();
	    if( tmpPhysDataNum!=0 ){
	      tmpDatas[j]->SetupSweptPhysData(tmpPhysDataNum);
	    }
	  }else{
	    Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetBndMagPhysDataNum(tmpMaterial);
	    if( tmpPhysDataNum!=0 ){
	      tmpDatas[j]->SetupSweptPhysData(tmpPhysDataNum);
	    }
	  }
	}
      } // index1
    } // index0
  } // dir
}


void 
ComboFields_DefineCntr::
LocateMemeory_For_3DVertexPhysDatas()
{
  Standard_Size theGlobalVIndxVec[2];
  Standard_Size theGlobalVIndx;
  Standard_Size theLocalFIndx;

  TxSlab2D<Standard_Integer> theRgn = GetGridGeom()->GetZRGrid()->GetXtndRgn();

  Standard_Integer Dir0 = 0;
  Standard_Integer Dir1 = 1;
  
  for(Standard_Size index0 = theRgn.getLowerBound(Dir0); index0<=theRgn.getUpperBound(Dir0); index0++){
    theGlobalVIndxVec[Dir0] = index0;
    for(Standard_Size index1 = theRgn.getLowerBound(Dir1); index1<=theRgn.getUpperBound(Dir1); index1++){
      theGlobalVIndxVec[Dir1] = index1;
      
      GetGridGeom()->GetZRGrid()->FillVertexIndx(theGlobalVIndxVec, theGlobalVIndx);

      Standard_Size faceSize= GetGridGeom_Cyl3D()->GetDimPhi();
      for(Standard_Size index2 = 0; index2 < faceSize; index2 ++){

      GridVertexData* tmpGridVertexData = GetGridGeom_Cyl3D()->GetGridGeometry(index2)->GetGridVertices()+theGlobalVIndx;

      Standard_Integer tmpMaterial = tmpGridVertexData->GetEMMaterialType();
      if( ! (GetFieldsDefineRules()->IsBndElecMaterial(tmpMaterial)) ){
	Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetCntrElecPhysDataNum();
	tmpGridVertexData->SetupSweptPhysData(tmpPhysDataNum);
      }else{
	Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetBndElecPhysDataNum(tmpMaterial);
	if( tmpPhysDataNum!=0 ){
	  tmpGridVertexData->SetupSweptPhysData(tmpPhysDataNum );
	}
      }

     }//index2
    } // index1
  } // index0
}


void 
ComboFields_DefineCntr::
LocateMemeory_For_3DFacePhysDatas()
{
  Standard_Size theGlobalFIndxVec[2];
  Standard_Size theGlobalFIndx;

  TxSlab2D<Standard_Integer> theRgn = GetGridGeom()->GetZRGrid()->GetXtndRgn();

  Standard_Integer Dir0 = 0;
  Standard_Integer Dir1 = 1;
  
  for(Standard_Size index0 = theRgn.getLowerBound(Dir0); index0<theRgn.getUpperBound(Dir0); index0++){
    theGlobalFIndxVec[Dir0] = index0;
    for(Standard_Size index1 = theRgn.getLowerBound(Dir1); index1<theRgn.getUpperBound(Dir1); index1++){
      theGlobalFIndxVec[Dir1] = index1;
      
      GetGridGeom()->GetZRGrid()->FillFaceIndx(theGlobalFIndxVec, theGlobalFIndx);

      Standard_Size faceSize= GetGridGeom_Cyl3D()->GetDimPhi();
      for(Standard_Size index2 = 0; index2 < faceSize; index2 ++){
      GridFace* tmpGridFace = (GetGridGeom_Cyl3D()->GetGridGeometry(index2))->GetGridFaces()+theGlobalFIndx;


      const vector<GridFaceData*>& tmpDatas = tmpGridFace->GetFaces();
      if(tmpDatas.empty()) continue;

      Standard_Size nb = tmpDatas.size();
      for(Standard_Size j=0;j<nb;j++){
	Standard_Integer tmpMaterial = tmpDatas[j]->GetEMMaterialType();
	if( ! (GetFieldsDefineRules()->IsBndMagMaterial(tmpMaterial)) ){
	  Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetCntrMagPhysDataNum();
	  tmpDatas[j]->SetupPhysData(tmpPhysDataNum);
	}else{
	  Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetBndMagPhysDataNum(tmpMaterial);
	  if( tmpPhysDataNum!=0 ){
	    tmpDatas[j]->SetupPhysData( tmpPhysDataNum );
	  }
	}
      }
     
     }//index2
    } // index1
  } // index0
}


void 
ComboFields_DefineCntr::
LocateMemeory_For_3DEdgePhysDatas()
{
  Standard_Size theGlobalEIndxVec[2];
  Standard_Size theGlobalEIndx;

  TxSlab2D<Standard_Integer> theRgn = GetGridGeom()->GetZRGrid()->GetXtndRgn();

  for(Standard_Integer dir=0; dir<2; dir++){
    Standard_Integer Dir0 = dir;
    Standard_Integer Dir1 = (Dir0+1)%2;

    for(Standard_Size index0 = theRgn.getLowerBound(Dir0); index0<theRgn.getUpperBound(Dir0); index0++){
      theGlobalEIndxVec[Dir0] = index0;
      for(Standard_Size index1 = theRgn.getLowerBound(Dir1); index1<=theRgn.getUpperBound(Dir1); index1++){
	theGlobalEIndxVec[Dir1] = index1;
	  
	GetGridGeom()->GetZRGrid()->FillEdgeIndx(Dir0, theGlobalEIndxVec, theGlobalEIndx);

      Standard_Size faceSize= GetGridGeom_Cyl3D()->GetDimPhi();
      for(Standard_Size index2 = 0; index2 < faceSize; index2 ++){
	GridEdge* tmpGridEdge = GetGridGeom_Cyl3D()->GetGridGeometry(index2)->GetGridEdges()[Dir0]+theGlobalEIndx;
	
	const vector<GridEdgeData*>& tmpDatas = tmpGridEdge->GetEdges();
	Standard_Size nb = tmpDatas.size();

	for(Standard_Size j=0;j<nb;j++){
	  Standard_Integer tmpMaterial = tmpDatas[j]->GetEMMaterialType();
	  if( ! (GetFieldsDefineRules()->IsBndElecMaterial(tmpMaterial)) ){
	    Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetCntrElecPhysDataNum();
	    if( tmpPhysDataNum!=0 ){
	      tmpDatas[j]->SetupPhysData(tmpPhysDataNum);
	    }
	  }else{
	    Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetBndElecPhysDataNum(tmpMaterial);
	    if( tmpPhysDataNum!=0 ){
	      tmpDatas[j]->SetupPhysData( tmpPhysDataNum );
	    }
	  }

	  if( ! (GetFieldsDefineRules()->IsBndMagMaterial(tmpMaterial)) ){
	    Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetCntrMagPhysDataNum();
	    if( tmpPhysDataNum!=0 ){
	      tmpDatas[j]->SetupSweptPhysData(tmpPhysDataNum);
	    }
	  }else{
	    Standard_Integer tmpPhysDataNum = GetFieldsDefineRules()->GetBndMagPhysDataNum(tmpMaterial);
	    if( tmpPhysDataNum!=0 ){
	      tmpDatas[j]->SetupSweptPhysData(tmpPhysDataNum);
	    }
	  }
	}
       }//idnex2
      } // index1
    } // index0
  } // dir
}
