// Standard includes
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
// std includes
#include <vector>

// io includes
#include <Hdf5Output.hxx>

#ifdef HAVE_HDF5

// Constructor

using namespace std;

Hdf5Output::Hdf5Output(string bName, size_t sq)
  : Hdf5IOBase(bName, sq)
{

}


Hdf5Output::~Hdf5Output()
{

}



void Hdf5Output::appendAttrib(const hid_t& targetId, const std::string &type,  const std::string &name) 
{
  // Add attribute that saves a scalar
  hid_t attr_DS = H5Screate(H5S_SCALAR);

  // Setup string type
  hid_t string_id = H5Tcopy(H5T_C_S1);
  size_t name_Length = strlen(name.c_str());
  H5Tset_size(string_id, name_Length);

  // Create attribute to store the num of elements
  hid_t attr_Id = H5Acreate(targetId, type.c_str(), string_id, attr_DS, H5P_DEFAULT, H5P_DEFAULT);

  // Write out the attribute
  H5Awrite( attr_Id, string_id, name.c_str() );
  // Close the attribute
  H5Aclose(attr_Id);
  // Close the attribute dataspace
  H5Sclose(attr_DS);
}


void Hdf5Output::appendAttrib(const hid_t& targetId, const std::string &name,  const size_t &value)
{
  // Add attribute that saves a scalar
  hid_t attr_DS = H5Screate(H5S_SCALAR);
  // Create attribute to store the num of elements
  hid_t attr_Id =  H5Acreate(targetId, name.c_str(), H5T_NATIVE_HSIZE, attr_DS, H5P_DEFAULT, H5P_DEFAULT);
  // Write out the attribute
  H5Awrite(attr_Id, H5T_NATIVE_HSIZE, (Standard_Real*) &value);
  // Close the attribute
  H5Aclose(attr_Id);
  // Close the attribute dataspace
  H5Sclose(attr_DS);
}


void Hdf5Output::appendAttrib(const hid_t& targetId, const std::string &name,  const int &value)
{
  // Add attribute that saves a scalar
  hid_t attr_DS = H5Screate(H5S_SCALAR);
  // Create attribute to store the num of elements
  hid_t attr_Id =  H5Acreate(targetId, name.c_str(), H5T_NATIVE_INT, attr_DS, H5P_DEFAULT, H5P_DEFAULT);
  // Write out the attribute
  H5Awrite(attr_Id, H5T_NATIVE_INT, (Standard_Real*) &value);
  // Close the attribute
  H5Aclose(attr_Id);
  // Close the attribute dataspace
  H5Sclose(attr_DS);
}



void Hdf5Output::appendAttrib(const hid_t& targetId, const std::string &name, const Standard_Real &x) 
{
  // Add attribute that saves a scalar
  hid_t attr_DS = H5Screate(H5S_SCALAR);
  // Create attribute to store the num of elements
  hid_t attr_Id =  H5Acreate(targetId, name.c_str(), this->h5FloatType, attr_DS, H5P_DEFAULT, H5P_DEFAULT);
  // Write out the attribute
  H5Awrite(attr_Id, this->h5FloatType, (Standard_Real*) &x);
  // Close the attribute
  H5Aclose(attr_Id);
  // Close the attribute dataspace
  H5Sclose(attr_DS);
}



void Hdf5Output::appendAttrib(const hid_t& targetId, const std::string &name, const std::vector<Standard_Real> &v)
{
  // For storing vectors
  size_t vSize = v.size();
  // FLOATTYPE vec[vSize];
  Standard_Real* vec = new Standard_Real[vSize];
  
  // Set up size of attribute
  hsize_t numDim = vSize;
  hid_t attr_DS = H5Screate_simple(1, &numDim, NULL);
  // Create attribute to store the number of elements
  hid_t attr_Id = H5Acreate(targetId, name.c_str(), this->h5FloatType, attr_DS, H5P_DEFAULT, H5P_DEFAULT);
  // Transfer to hdf5 form and write
  for (size_t i = 0; i < vSize; ++i) vec[i] = v[i];
  H5Awrite(attr_Id, this->h5FloatType, vec);
  // Close the attribute
  H5Aclose(attr_Id);
  // Close the attribute dataspace
  H5Sclose(attr_DS);
  
  delete[] vec;
}


void Hdf5Output::appendAttrib(const hid_t& targetId, const std::string &name, const std::vector<Standard_Integer> &v) 
{
  // For storing vectors
  size_t vSize = v.size();
  int* vec = new int[vSize];
  
  // Set up size of attribute
  hsize_t numDim = vSize;
  hid_t attr_DS = H5Screate_simple(1, &numDim, NULL);
  // Create attribute to store the number of elements
  hid_t attr_Id =  H5Acreate(targetId, name.c_str(), H5T_NATIVE_INT, attr_DS, H5P_DEFAULT, H5P_DEFAULT);
  // Transfer to hdf5 form and write
  for (size_t i = 0; i < vSize; ++i) vec[i] = v[i];
  H5Awrite(attr_Id, H5T_NATIVE_INT, vec);
  // Close the attribute
  H5Aclose(attr_Id);
  // Close the attribute dataspace
  H5Sclose(attr_DS);
  delete[] vec;
}





#endif
