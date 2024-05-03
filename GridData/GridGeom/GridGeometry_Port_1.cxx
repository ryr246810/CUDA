#include <GridGeometry.hxx>
#include <PortDataFunc.hxx>
#include <stdlib.h>

//#define NONPMLPORT_DBG

bool 
GridGeometry::
Does_NonPMLPortRgn_Overlap_With_PMLRgn(const PortData& thePort)
{
  bool result = false;

  if( !IsPMLPortType(thePort.m_Type)){
    TxSlab2D<Standard_Integer> thePortRgn;
    ComputeAbsorbingRgnAccordingOpenPort(thePort, thePortRgn);

#ifdef NONPMLPORT_DBG
      cout<<"Does_NonPMLPortRgn_Overlap_With_PMLRgn------------PortRgn----------------------------------------------->>>"<<endl;
      thePortRgn.write(cout);
      cout<<"Does_NonPMLPortRgn_Overlap_With_PMLRgn------------PortRgn-----------------------------------------------<<<"<<endl;
#endif

    Standard_Integer theMidPntGlobalIndx[2];
    Standard_Integer theShiftedMidPntGlobalIndx[2];
    
    Standard_Integer dir0 = thePort.m_Dir;
    Standard_Integer dir1 = (dir0+1)%2;
    
    if(thePort.m_RelativeDir==1){
      theMidPntGlobalIndx[dir0] = thePortRgn.getUpperBound(dir0); 
    }else{
      theMidPntGlobalIndx[dir0] = thePortRgn.getLowerBound(dir0); 
    }
    
    theMidPntGlobalIndx[dir1] = 0.5 * (thePortRgn.getLowerBound(dir1) + thePortRgn.getUpperBound(dir1));

    theShiftedMidPntGlobalIndx[dir0] = theMidPntGlobalIndx[dir0];
    theShiftedMidPntGlobalIndx[dir1] = theMidPntGlobalIndx[dir1] + 1;
    
    TxSlab2D<Standard_Integer> theDgnRgn(theMidPntGlobalIndx, theShiftedMidPntGlobalIndx);
    vector<GridEdgeData*>  theDgnGridDatas;
    this->GetGridEdgeDatasNotOfMaterialTypeOfSubRgn(Standard_Integer(PEC), theDgnRgn, false, theDgnGridDatas);

    for(Standard_Size ne = 0; ne<theDgnGridDatas.size(); ne++){
      if(theDgnGridDatas[ne]->IsMaterialType(PML)){
	result = true;
	break;
      }
    }
  }
  return result;
}


void 
GridGeometry::
SetNonPMLPortBnd()
{
#ifdef NONPMLPORT_DBG
  cout<<"SetNonPMLPortBnd------------GlobalRgn---------------------------------------------------->>>"<<endl;
  TxSlab2D<Standard_Integer> theGGRgn;
  this->GetZRGrid()->GetXtndRgn(theGGRgn);
  theGGRgn.write(cout);
  cout<<"SetNonPMLPortBnd------------GlobalRgn----------------------------------------------------<<<"<<endl;
#endif

  const map<Standard_Integer, PortData, less<Standard_Integer> >* thePorts = this->GetGridBndDatas()->GetPorts();
  map<Standard_Integer, PortData, less<Standard_Integer> >::const_iterator iter;

  for(iter = thePorts->begin(); iter!=thePorts->end(); iter++){

    Standard_Integer thePortIndex = iter->first;
    const PortData& thePort = iter->second;
    Standard_Integer thePortDir = thePort.m_Dir;
    Standard_Integer theRelativeDir = thePort.m_RelativeDir;

    if(IsPMLPortType(thePort.m_Type)){
      continue;
    }

    if( Does_NonPMLPortRgn_Overlap_With_PMLRgn(thePort) ) {
      cout<<"GridGeometry::SetNonPMLPortBnd------------------------port are not set correctly"<<endl;
      exit(1);
    }else{
      TxSlab2D<Standard_Integer> theMurRgn;
      ComputeAbsorbingRgnAccordingOpenPort(thePort, theMurRgn);

      // 1.0 get the all GridEdgeDatas whose material are not PEC
      vector<GridEdgeData*>  theEdgeDatas;
      GetGridEdgeDatasNotOfMaterialTypeOfSubRgn( Standard_Integer(PEC),theMurRgn, false, theEdgeDatas);
      
#ifdef NONPMLPORT_DBG
      theMurRgn.write(cout);
      cout<<"SetNonPMLPortBnd---------------theEdgeDatas.size()\t=\t"<<theEdgeDatas.size()<<endl;
#endif

      for(Standard_Size ne = 0; ne<theEdgeDatas.size(); ne++){
	theEdgeDatas[ne]->SetMaterialType(MUR);
      }
      theEdgeDatas.clear();

      vector<GridFaceData*>  theFaceDatas;
      GetGridFaceDatasNotOfMaterialTypeOfSubRgn( Standard_Integer(PEC),theMurRgn, theFaceDatas);

#ifdef NONPMLPORT_DBG
      theMurRgn.write(cout);
      cout<<"SetNonPMLPortBnd---------------theFaceDatas.size()\t=\t"<<theFaceDatas.size()<<endl;
#endif

      for(Standard_Size nf = 0; nf<theFaceDatas.size(); nf++){
	theFaceDatas[nf]->SetMaterialType(MUR);
      }
      theFaceDatas.clear();


      vector<GridVertexData*>  theVertexDatas;
      GetGridVertexDatasNotOfMaterialTypeOfSubRgn( Standard_Integer(PEC),theMurRgn, false, theVertexDatas);

#ifdef NONPMLPORT_DBG
      theMurRgn.write(cout);
      cout<<"SetNonPMLPortBnd---------------theVertexDatas.size()\t=\t"<<theVertexDatas.size()<<endl;
#endif

      for(Standard_Size nv = 0; nv<theVertexDatas.size(); nv++){
	theVertexDatas[nv]->SetMaterialType(MUR);
      }
      theVertexDatas.clear();

    } //if(!Does_NonPMLPortRgn_Overlap_With_PMLRgn)
  } // for(iter=......)
}
