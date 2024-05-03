
#ifndef _DynObjBase_HeaderFile
#define _DynObjBase_HeaderFile

// std includes
#include <string>


/*** Base class for dynamic objects **/
class DynObjBase 
{

  public:  

  /*** Constructor - does nothing. */
  DynObjBase(){}
  
  /*** Constructor - does nothing. */
  DynObjBase(std::string nm){
    m_Name = nm;
  }

  /*** Destructor */
  virtual ~DynObjBase(){}
  
  /*** set the base name for a file dump  */
  virtual void SetName(const std::string& nm) { m_Name = nm; }
  
  /*** get the base name*/
  virtual std::string GetName() const { return m_Name; }
  
  /**
   * update system
   *
   * @param time the new time to update to
   */
  virtual void Update(double time) = 0;


 protected:
  /*** The name for this object */
  std::string m_Name;
  
};

#endif 
