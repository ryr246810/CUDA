// -----------------------------------------------------------------------
// File:	Hdf5IOBase.hxx
// Purpose:	Class that handles creating, writing hdf5 format data files.
// -----------------------------------------------------------------------

#ifndef _Hdf5IOBase_HeaderFile
#define _Hdf5IOBase_HeaderFile


#ifdef HAVE_HDF5

// std includes
#include <string>

// This includes config.h
#include <hdf5.h>

#include <Standard_TypeDefine.hxx>


#define HDF5_FAIL -1

class Hdf5IOBase
{
public:
  Hdf5IOBase(std::string baseName, size_t seqNumber);
  virtual ~Hdf5IOBase();

protected:
  /** The sequence number */
  size_t seqNumber;
  
  /** The base name for the data files.  */
  std::string baseName;
  
  /** The file extension for the data files */
  std::string dataFileExt;

  /*** The hdf5 floattype to be used in the data file */
  hid_t h5FloatType;
  hid_t h5SizeType;  

  /** The buffer for writing data */
  mutable hsize_t bufSize;

  /** The buffer for writing data */
  mutable Standard_Real* buffer;


private:
  Hdf5IOBase(); 
  // Make the usual suspects private to prevent use.
  Hdf5IOBase(const Hdf5IOBase&);
  Hdf5IOBase& operator=(const Hdf5IOBase&);
};

#endif // HAVE_HDF5

#endif // _Hdf5IOBase_HeaderFile
