
// -----------------------------------------------------------------------
// File:        FldHdf5Output_0.cpp
// Purpose:     Class that handles creating, writing hdf5 format data files.
// -----------------------------------------------------------------------

// Standard includes
#include <unistd.h>
#include <time.h>
#include <assert.h>

// std includes
#include <vector>


// vpbase includes
#include <NodeFldsBase.hxx>

// vpio includes
#include <NodeFlds_Hdf5Output.hxx>

#include <CFLDSSchema.hxx>

#ifdef HAVE_HDF5
#undef DEBUG_IO


void NodeFlds_Hdf5Output::appendFieldAttribs()
{
  std::string type = CFLDSSchema::centeringAtt;
  std::string name = CFLDSSchema::nodalCenteringKey;
  appendAttrib(fieldDataSetId, type,  name);

  type = CFLDSSchema::meshKey;
  name = "globalGrid";
  appendAttrib(fieldDataSetId, type,  name);

  type = CFLDSSchema::typeAtt;
  name = CFLDSSchema::varKey;
  appendAttrib(fieldDataSetId, type,  name);
}





void NodeFlds_Hdf5Output::appendFieldDerivedVariablesAttrib() 
{
  if(getField()->GetElementNum()==3){
    hid_t derivedVariablesId = H5Gcreate(fieldGroupId, "derivedVariables", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(derivedVariablesId != HDF5_FAIL);
    
    std::string type = getField()->GetName();
    
    std::string name = "{" + getField()->GetName() + "_0," + getField()->GetName() + "_1," + getField()->GetName() + "_2}";
    appendAttrib(derivedVariablesId, type,  name);
    
    type = CFLDSSchema::typeAtt;
    name = CFLDSSchema::varsKey;
    appendAttrib(derivedVariablesId, type,  name);
    
    H5Gclose(derivedVariablesId);
  }
}



void NodeFlds_Hdf5Output::appendFieldRunInforAttrib()
{
  hid_t runInforId = H5Gcreate(fieldGroupId, (CFLDSSchema::runInfoKey).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  assert(runInforId != HDF5_FAIL);

  std::string type = CFLDSSchema::typeAtt;
  std::string name = CFLDSSchema::runInfoKey;
  appendAttrib(runInforId, type,  name);

  H5Gclose(runInforId);
}



void NodeFlds_Hdf5Output::appendFieldglobalGridAttrib()
{
  hid_t globalGridGlobalId = H5Gcreate(fieldGroupId, "globalGrid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  assert(globalGridGlobalId != HDF5_FAIL);

  /*
  TxSlab2D<Standard_Integer> fldsRgn = this->getField()->GetZRGrid()->GetPhysRgn();
  TxSlab2D<Standard_Integer> orgPhysRgn = fldsRgn;
  fldsRgn.shiftBack(orgPhysRgn);
  //*/

  TxSlab2D<Standard_Integer> fldsRgn  = this->getField()->GetZRGrid()->GetXtndRgn();
  TxSlab2D<Standard_Real> physRealRgn = this->getField()->GetZRGrid()->GetRealRgn();


  std::string type = CFLDSSchema::typeAtt ;
  std::string name = CFLDSSchema::meshKey;
  appendAttrib(globalGridGlobalId, type,  name);


  type = CFLDSSchema::kindAtt;
  name = CFLDSSchema::Uniform::key;
  appendAttrib(globalGridGlobalId, type,  name);


  type = CFLDSSchema::Uniform::lowerBounds;
  vector<Standard_Real> lowerBounds;
  lowerBounds.clear();
  lowerBounds.push_back(physRealRgn.getLowerBound(0));
  lowerBounds.push_back(physRealRgn.getLowerBound(1));
  lowerBounds.push_back(physRealRgn.getLowerBound(2));
  appendAttrib(globalGridGlobalId, type, lowerBounds);


  type = CFLDSSchema::Uniform::numCells;
  vector<Standard_Integer> dims;
  dims.clear();
  dims.push_back(fldsRgn.getLength(0));
  dims.push_back(fldsRgn.getLength(1));
  dims.push_back(fldsRgn.getLength(2));
  appendAttrib(globalGridGlobalId, type, dims);


  type = CFLDSSchema::Uniform::startCell;
  vector<Standard_Integer> startCell;
  startCell.clear();
  startCell.push_back(fldsRgn.getLowerBound(0));
  startCell.push_back(fldsRgn.getLowerBound(1));
  startCell.push_back(fldsRgn.getLowerBound(2));
  appendAttrib(globalGridGlobalId, type, startCell);


  type = CFLDSSchema::Uniform::upperBounds;
  vector<Standard_Real> upperBounds;
  upperBounds.clear();
  upperBounds.push_back(physRealRgn.getUpperBound(0));
  upperBounds.push_back(physRealRgn.getUpperBound(1));
  upperBounds.push_back(physRealRgn.getUpperBound(2));
  appendAttrib(globalGridGlobalId, type, upperBounds);


  H5Gclose(globalGridGlobalId);
}



void NodeFlds_Hdf5Output::appendFieldTimeAttrib()
{
  hid_t timeId = H5Gcreate(fieldGroupId, (CFLDSSchema::timeKey).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  assert(timeId != HDF5_FAIL);

  std::string type = CFLDSSchema::typeAtt;
  std::string name = CFLDSSchema::timeKey;
  appendAttrib(timeId, type,  name);

  type = CFLDSSchema::timeAtt;
  double time = getField()->GetCurTime();
  appendAttrib(timeId, type, time);

  H5Gclose(timeId);
}


#endif // HAVE_HDF5
