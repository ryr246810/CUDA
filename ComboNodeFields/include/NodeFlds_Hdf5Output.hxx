// -----------------------------------------------------------------------
// File:	NodeFlds_Hdf5Output.hxx
// Purpose:	Class that handles creating, writing hdf5 format data files.
// -----------------------------------------------------------------------

#ifndef _NodeFldsHdf5Output_HeaderFile
#define _NodeFldsHdf5Output_HeaderFile

#ifdef HAVE_HDF5

// std includes
#include <string>

// This includes config.h
#include <hdf5.h>

// vpbase includes
#include <Hdf5Output.hxx>
#include <NodeFlds_OutputBase.hxx>
#include <Standard_TypeDefine.hxx>

#include <NodeFlds_BufferWriter.hxx>
#include <Standard_TypeDefine.hxx>


class NodeFlds_Hdf5Output : public NodeFlds_OutputBase, public Hdf5Output
{
public:
  NodeFlds_Hdf5Output(std::string baseName, size_t seqNumber);
  virtual ~NodeFlds_Hdf5Output();

public:
  /*** Creates an output file for a field   */
  virtual void createFldFile();

  /*** Creates field data for field file  */
  virtual void createFieldData();

  /*** Writes data to field file  */
  virtual void writeField();

  /*** closes the field data */
  virtual void closeFieldData();

  /*** closes the field file */
  virtual void closeFieldFile();

  virtual void appendFieldAttribs();


  virtual void appendFieldDerivedVariablesAttrib();
  virtual void appendFieldglobalGridAttrib();
  virtual void appendFieldRunInforAttrib();
  virtual void appendFieldTimeAttrib();


protected:
  /** The buffer writer */
  mutable NodeFlds_BufferWriter bufferWriter;


private:
  /*** List of fields that the dataWriter has dumped */
  vector<std::string> fieldNames;

  /**************************************************************/
  TxSlab2D<int> dataRgnSlab;

  /*** The start of the hyperslab for writing/reading from the full data file  */
  mutable hsize_t fieldStart[3];

  /*** The extent of the hyperslab for writing/reading from the full data file  */
  mutable hsize_t fieldExtent[3];



  /********************** Field Data*********************/
  /** Field file ID */
  hid_t fieldFileId;
  
  /** Field group ID */
  hid_t fieldGroupId;
  
  /** Field dataset ID */
  hid_t fieldDataSetId;
  
  /** Field file space ID (all the data) */
  hid_t fieldFileSpaceId;
  
  /** Field data space ID (data from this processor) */
  hid_t fieldDataSpaceId;
  
  /** Field memspace ID */
  hid_t fieldMemSpaceId;


private:
  NodeFlds_Hdf5Output();
  // Make copy constructor and assignment operator private to prevent their use
  NodeFlds_Hdf5Output(const NodeFlds_Hdf5Output&);
  NodeFlds_Hdf5Output& operator=(const NodeFlds_Hdf5Output&);
};

#endif // HAVE_HDF5

#endif
