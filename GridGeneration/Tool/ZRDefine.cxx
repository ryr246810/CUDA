
#include <ZRDefine.hxx>


ZRDefine::
ZRDefine()
{

}


ZRDefine::
ZRDefine(Standard_Real theXYZOrg[3], Standard_Integer zdir, Standard_Integer rdir)
{
  m_XYZOrg = TxVector<Standard_Real>(theXYZOrg[0], theXYZOrg[1], theXYZOrg[2]);
  m_ZDir = zdir;
  m_RDir = rdir;
  m_WorkPlanDir = 3-m_ZDir-m_RDir;

  m_WorkPlaneUnitVec = TxVector<Standard_Real>(0.0, 0.0, 0.0);
  m_ZUnitVec = TxVector<Standard_Real>(0.0, 0.0, 0.0);
  m_RUnitVec = TxVector<Standard_Real>(0.0, 0.0, 0.0);

  m_WorkPlaneUnitVec[m_WorkPlanDir] = 1.0;
  m_ZUnitVec[m_ZDir] = 1.0;
  m_RUnitVec[m_RDir] = 1.0;
}


ZRDefine::
~ZRDefine()
{

}


Standard_Integer 
ZRDefine::
GetZDir() const
{
  return m_ZDir;
}

Standard_Integer 
ZRDefine::
GetRDir() const
{
  return m_RDir;
}

Standard_Integer 
ZRDefine::
GetWorkPlaneDir() const
{
  return m_WorkPlanDir;
}

Standard_Integer 
ZRDefine::
GetZRDir_AccordingTo_XYZDir(const Standard_Integer aDir) const
{
  Standard_Integer result = 0;
  if(aDir==m_ZDir){
    result = 0;
  }else if(aDir==m_RDir){
    result = 1;
  }else{
    result = 2;
  }
  return result;
}


Standard_Integer 
ZRDefine::
GetXYZDir_AccordingTo_ZRDir(const Standard_Integer aDir) const
{
  Standard_Integer result = 0;
  switch (aDir)
    {
    case 0:
      {
	result=m_ZDir;
	break;
      }
    case 1:
      {
	result=m_RDir;
	break;
      }
    case 2:
      {
	result=m_WorkPlanDir;
	break;
      }
    }
  return result;
}


const TxVector<Standard_Real>& 
ZRDefine::
GetZUnitVec() const
{
  return m_ZUnitVec;
}

const TxVector<Standard_Real>& 
ZRDefine::
GetRUnitVec() const
{
  return m_RUnitVec;
}

const TxVector<Standard_Real>& 
ZRDefine::
GetWorkPlaneUnitVec() const
{
  return m_WorkPlaneUnitVec;
}

const TxVector<Standard_Real>& 
ZRDefine::
GetXYZUnitVecAccordingRZDir(Standard_Integer dir) const
{
  if(dir==0){
    return m_ZUnitVec;
  }else if(dir==1){
    return m_RUnitVec;
  }else{
    return m_WorkPlaneUnitVec;
  }
}

const TxVector<Standard_Real>& 
ZRDefine::
GetXYZOrg() const
{
  return m_XYZOrg;
}



void 
ZRDefine::
Convert_XYZ_to_ZR(const TxSlab<Standard_Real>& theXYZSlab, TxSlab2D<Standard_Real>& theZRSlab) const
{
  theZRSlab.setBounds(theXYZSlab.getLowerBound(m_ZDir), 
		      theXYZSlab.getLowerBound(m_RDir), 
		      theXYZSlab.getUpperBound(m_ZDir), 
		      theXYZSlab.getUpperBound(m_RDir));
}


void 
ZRDefine::
Convert_ZR_to_XYZ(const Standard_Real theZRPnt[2], TxVector<Standard_Real>& theXYZ) const
{
  theXYZ[m_ZDir] = theZRPnt[0];
  theXYZ[m_RDir] = theZRPnt[1];
  theXYZ[m_WorkPlanDir] = m_XYZOrg[m_WorkPlanDir];
}



void 
ZRDefine::
Convert_ZR_to_XYZ(const TxVector2D<Standard_Real>& theZRPnt, TxVector<Standard_Real>& theXYZ) const
{
  theXYZ[m_ZDir] = theZRPnt[0];
  theXYZ[m_RDir] = theZRPnt[1];
  theXYZ[m_WorkPlanDir] = m_XYZOrg[m_WorkPlanDir];
}



void 
ZRDefine::
Convert_XYZ_to_ZR(const TxVector<Standard_Real>& theXYZ, Standard_Real theZRPnt[2]) const
{
  theZRPnt[0] = theXYZ[m_ZDir];
  theZRPnt[1] = theXYZ[m_RDir];
}


void 
ZRDefine::
Convert_XYZ_to_ZR(const TxVector<Standard_Real>& theXYZ, TxVector2D<Standard_Real>& theZRPnt) const
{
  theZRPnt[0] = theXYZ[m_ZDir];
  theZRPnt[1] = theXYZ[m_RDir];
}
