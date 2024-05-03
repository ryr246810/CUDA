#include <GridBndDefine.hxx>
#include <PortDataFunc.hxx>
#include <string>
#include <sstream>


string CompoundName(string baseName, int index)
{
  stringstream sstr;
  sstr<<baseName;
  sstr<<index;
  return sstr.str();
}


string CompoundName(string baseName, int index, string fileExt)
{
  stringstream sstr;
  sstr<<baseName;
  sstr<<index;
  sstr<<fileExt;
  return sstr.str();
}




int Get_PortRgn_To_PMLRgn_Gap()
{
  return 4;
}



void ComputeAbsorbingRgnAccordingOpenPort(const PortData& thePort, TxSlab2D<Standard_Integer>& theRgn)
{
  theRgn.setBounds(thePort.m_LDCords[0], thePort.m_LDCords[1], 
		   thePort.m_RUCords[0], thePort.m_RUCords[1]);

  if(thePort.m_RelativeDir==1){
    theRgn.setLowerBound(thePort.m_Dir, (thePort.m_LDCords[thePort.m_Dir]+1));
  }else{
    theRgn.setUpperBound(thePort.m_Dir, (thePort.m_RUCords[thePort.m_Dir]-1));
  }
}



void ComputePMLInfAccordingOpenPort(const PortData& thePort,
				    TxSlab2D<Standard_Integer>& theRgn,
				    Standard_Integer& theStartIndex,
				    Standard_Integer& theLayerNum)
{
  theRgn.setBounds(thePort.m_LDCords[0], thePort.m_LDCords[1], 
		   thePort.m_RUCords[0], thePort.m_RUCords[1]);

  if(thePort.m_RelativeDir==1){
    if(IsInputPMLPortType(thePort.m_Type)){
      theStartIndex = thePort.m_LDCords[ thePort.m_Dir ] + Get_PortRgn_To_PMLRgn_Gap() + 1;
    }else{
      theStartIndex = thePort.m_LDCords[ thePort.m_Dir ] + 1;
    }
    theRgn.setLowerBound(thePort.m_Dir, theStartIndex);
  }else{
   if(IsInputPMLPortType(thePort.m_Type)){
     theStartIndex = thePort.m_RUCords[ thePort.m_Dir ] - Get_PortRgn_To_PMLRgn_Gap() - 1;
   }else{
     theStartIndex = thePort.m_RUCords[ thePort.m_Dir ] - 1;
   }
    theRgn.setUpperBound(thePort.m_Dir, theStartIndex);
  }

  theLayerNum = theRgn.getLength(thePort.m_Dir);
}


void ComputePortStartIndex(const PortData& thePort, Standard_Integer& theStartIndex)
{
  if( thePort.m_RelativeDir==1){
    theStartIndex = thePort.m_LDCords[ thePort.m_Dir ];
  }else{
    theStartIndex = thePort.m_RUCords[ thePort.m_Dir ];
  }
}


void ComputeMurTypePortRgn(const PortData& thePort, TxSlab2D<Standard_Integer>& theMurPortRgn)
{
  theMurPortRgn.setBounds(thePort.m_LDCords[0], thePort.m_LDCords[1], 
			  thePort.m_RUCords[0], thePort.m_RUCords[1]);

  Standard_Integer theInterfaceGlobalIndx;

  if(thePort.m_RelativeDir==1){
    theInterfaceGlobalIndx = thePort.m_LDCords[ thePort.m_Dir ];
  }else{
    theInterfaceGlobalIndx = thePort.m_RUCords[ thePort.m_Dir ];
  }

  Standard_Integer thePortDir = thePort.m_Dir;

  if(thePort.m_RelativeDir==1){
    theMurPortRgn.setLowerBound(thePort.m_Dir, (theInterfaceGlobalIndx+1));
    theMurPortRgn.setUpperBound(thePort.m_Dir, (theInterfaceGlobalIndx+1));
  }else{
    theMurPortRgn.setLowerBound(thePort.m_Dir, (theInterfaceGlobalIndx-1));
    theMurPortRgn.setUpperBound(thePort.m_Dir, (theInterfaceGlobalIndx-1));
  }
}


void ComputePortPhysStartIndex(const PortData& thePort, Standard_Integer& theInterfaceIndex)
{
  if( thePort.m_RelativeDir==1){
    theInterfaceIndex = thePort.m_LDCords[ thePort.m_Dir ] + 1;
  }else{
    theInterfaceIndex = thePort.m_RUCords[ thePort.m_Dir ] - 1;
  }
}
