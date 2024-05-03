// -----------------------------------------------------------------------
// File:        NodeFlds_Hdf5Output.cpp
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


#ifdef HAVE_HDF5
#undef DEBUG_IO

// Constructor

NodeFlds_Hdf5Output::NodeFlds_Hdf5Output(std::string bName,  size_t sq)
:Hdf5Output(bName, sq) 
{
  // set up the file extensions
  stringstream sstr;
  sstr << "_" << seqNumber << ".cflds";
  sstr >> dataFileExt;
}


NodeFlds_Hdf5Output::~NodeFlds_Hdf5Output() {
}


void NodeFlds_Hdf5Output::createFldFile()
{
  // Add the field name to list
  fieldNames.push_back(getField()->GetName());

  // Set up parallel access properties
  std::string fileName = this->baseName + "_" + getField()->GetName();

  fileName += this->dataFileExt;
  fieldFileId = H5Fcreate(fileName.c_str(),
			  H5F_ACC_TRUNC, 
			  H5P_DEFAULT, 
			  H5P_DEFAULT);
  
  if (fieldFileId<0) {
    cerr << "Unable to create file '" 
	 << fileName.c_str() 
	 << "'.  Error # " 
	 << fieldFileId 
	 << endl;
    return;
  }
}


void NodeFlds_Hdf5Output::createFieldData()
{
  size_t NDIM = 2;
  //1. Create a group for the field in the hdf5 file-----an already existed root path
  fieldGroupId = H5Gopen(fieldFileId, "/", H5P_DEFAULT);
  assert(fieldGroupId != HDF5_FAIL);

  //2. Set up the dataspace for the field data
  hsize_t dataShape[NDIM+1];

  //2.1 Get appropriate rgn for dumping
  TxSlab2D<int> fldRgn = (getField()->GetZRGrid())->GetPhysRgn();

  //2.2 get length according rgn definition
  for (size_t i=0; i<NDIM; ++i) dataShape[i] = (fldRgn.getLength(i)+1);  //all node
  dataShape[NDIM] = getField()->GetElementNum();

  //2.3 define data space
  fieldFileSpaceId = H5Screate_simple(NDIM+1,
				      dataShape,
				      NULL);
  assert(fieldFileSpaceId != HDF5_FAIL);

  //3 Tag the field data with the field name
  std::string dataName = getField()->GetName();
  //3.1 Set up the dataset
  fieldDataSetId = H5Dcreate(fieldGroupId, dataName.c_str(), 
			     this->h5FloatType, fieldFileSpaceId, 
			     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  assert(fieldDataSetId != HDF5_FAIL);

  H5Sclose(fieldFileSpaceId);
}


void NodeFlds_Hdf5Output::writeField()
{
  herr_t ret;	// Return error

  dataRgnSlab = (getField()->GetZRGrid())->GetPhysRgn();
  fieldDataSpaceId = fieldMemSpaceId = H5S_ALL;

  hsize_t dataSize = 
    (dataRgnSlab.getLength(0)+1)*
    (dataRgnSlab.getLength(1)+1)*
    getField()->GetElementNum();

  if (dataSize > this->bufSize) {
    delete[] this->buffer;
    this->buffer = new Standard_Real[dataSize];
    this->bufSize = dataSize;
  }

  this->bufferWriter.setBuffer(this->buffer);
  this->bufferWriter.setField(getField());
  this->bufferWriter.setRegion(dataRgnSlab);

  this->bufferWriter.UpdateVertices();

  // Set up collective I/O prop list
  hid_t xferPropList = H5P_DEFAULT;

  // Write out the this->buffer 
  ret = H5Dwrite(fieldDataSetId, this->h5FloatType, fieldMemSpaceId, fieldDataSpaceId, xferPropList, this->buffer);
  assert(ret != HDF5_FAIL);
}


void NodeFlds_Hdf5Output::closeFieldData()
{
  // Close the dataset
  H5Dclose(fieldDataSetId);
  fieldDataSetId = 0;
  // Close the group
  H5Gclose(fieldGroupId);
  fieldGroupId = 0;
}


void NodeFlds_Hdf5Output::closeFieldFile() 
{
  // Close the data file and wait for all to finish
  H5Fclose(fieldFileId);
  fieldFileId = 0;
}


#endif // HAVE_HDF5
