// --------------------------------------------------------------------
//
// File:	FuncDefBase.h

// --------------------------------------------------------------------

#ifndef _FuncDefBase_HeaderFile
#define _FuncDefBase_HeaderFile

// txbase includes
#include <TxDblFormulaList.h>

#include <TxHierAttribSet.h>
#include <Standard_TypeDefine.hxx>


/**  
 * Base class for functors templated over precision.
 */
class FuncDefBase
{

public:  
  FuncDefBase();
  virtual ~FuncDefBase();
  
  /**
   * Set up the functor from data in an attribute set
   * 
   * @param tas A TxAttribute set containing the parameters of the
   * plane wave, which are:
   *
   * tas.getPrmVec("expression") the expression to be evaluated 
   */
  virtual void setAttrib_Param(const TxHierAttribSet& tas);
  virtual void setAttrib_DFL(const TxHierAttribSet& tas);
  virtual void combine_DFL_Params();

  static void setGlobalAttrib(const TxHierAttribSet& tas);
  static void GetGlobalVariables(std::map< std::string, double, std::less<std::string> >& theGlobalVariables);

 protected:
  mutable std::map< std::string, double, std::less<std::string> > m_params;
  TxDblFormulaList* m_dfl; 

  static std::map< std::string, double, std::less<std::string> > m_global_params;
  static TxDblFormulaList* m_global_dfl; 
};

#endif 
