// -----------------------------------------------------------------------
// File:	NodeFlds_OutputBase.hxx
// Purpose:	Abstract base class to provide interfaces for doing field-data output.
// -----------------------------------------------------------------------

#ifndef _NodeFlds_OutputBase_HeaderFile
#define _NodeFlds_OutputBase_HeaderFile

// std includes
#include <string>
#include <vector>

// TxBase includes
#include <TxStreams.h>

// vpbase includes
#include <Standard_TypeDefine.hxx>


class NodeFldsBase;

class NodeFlds_OutputBase
{
public:
  NodeFlds_OutputBase() {};

  void setField(const NodeFldsBase* _field){
    m_Field = _field;
  };

  const NodeFldsBase* getField() const{
    return m_Field;
  }

  const NodeFldsBase& getFieldRef() const{
    return *m_Field;
  }

  virtual ~NodeFlds_OutputBase() {};

  virtual void createFldFile() = 0;

  virtual void createFieldData() = 0;

  virtual void writeField() = 0;
  
  virtual void closeFieldData() = 0;
  
  virtual void closeFieldFile() = 0;

  virtual void appendFieldAttribs() = 0;
  virtual void appendFieldDerivedVariablesAttrib() = 0;
  virtual void appendFieldglobalGridAttrib() = 0;
  virtual void appendFieldRunInforAttrib() = 0;
  virtual void appendFieldTimeAttrib() = 0;

private:
  NodeFlds_OutputBase(const NodeFlds_OutputBase& field);
  NodeFlds_OutputBase& operator=(const NodeFlds_OutputBase& field);

private:
  const NodeFldsBase* m_Field;
};

#endif
