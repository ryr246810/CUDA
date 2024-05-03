// Standard includes
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
// std includes
#include <vector>

// io includes
#include <Hdf5IOBase.hxx>

#include <sstream>

#ifdef HAVE_HDF5

using namespace std;

// Constructor
Hdf5IOBase::Hdf5IOBase(string bName, size_t sq)
{
  // get the base name
  baseName = bName;

  // the sequence number
  seqNumber = sq;

  // set up the file extensions
  stringstream sstr;
  sstr << "_" << seqNumber << ".h5";
  sstr >> dataFileExt;

  // set the floattype
  h5FloatType = H5T_NATIVE_DOUBLE;
  h5SizeType  = H5T_NATIVE_HSIZE;

  // Set the buffer to zero size
  buffer = NULL;
  bufSize = 0;
}


Hdf5IOBase::~Hdf5IOBase()
{
// Delete the data
  if(buffer!=NULL)
    delete[] buffer;
}


#endif
