// -----------------------------------------------------------------------
// File:	Hdf5IO.hxx
// Purpose:	Class that handles creating, writing hdf5 format data files.
// -----------------------------------------------------------------------

#ifndef _Hdf5Output_HeaderFile
#define _Hdf5Output_HeaderFile


#ifdef HAVE_HDF5

// std includes
#include <string>

// This includes config.h
#include <hdf5.h>


#include <Hdf5IOBase.hxx>

#include <Standard_TypeDefine.hxx>


#define HDF5_FAIL -1

class Hdf5Output : public Hdf5IOBase
{
public:

  Hdf5Output(std::string baseName, size_t seqNumber);

  virtual ~Hdf5Output();

public:
  /*** Appends an attribute to target id */
  virtual void appendAttrib(const hid_t& targetId, const std::string &name,  const size_t &value) ;
  virtual void appendAttrib(const hid_t& targetId, const std::string &name,  const int &value) ;
  virtual void appendAttrib(const hid_t& targetId, const std::string &name,  const std::string &value) ;
  virtual void appendAttrib(const hid_t& targetId, const std::string &name,  const Standard_Real &x);
  virtual void appendAttrib(const hid_t& targetId, const std::string &name,  const std::vector<Standard_Real> &v);
  virtual void appendAttrib(const hid_t& targetId, const std::string &name,  const std::vector<Standard_Integer> &v);



private:
  // Make the usual suspects private to prevent use.
  Hdf5Output();
  Hdf5Output(const Hdf5Output&);
  Hdf5Output& operator=(const Hdf5Output&);
};

#endif // HAVE_HDF5

#endif // _Hdf5Output_HeaderFile
