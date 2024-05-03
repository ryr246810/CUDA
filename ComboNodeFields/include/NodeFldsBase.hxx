#ifndef _NodeFldsBase_HeaderFile
#define _NodeFldsBase_HeaderFile

// std includes
#include <string>


// EMPIC_Suit includes
#include <DynObj.hxx>

#include <GridGeometry.hxx>
#include <IndexAndWeights.hxx>

#include <NodeFlds_OutputBase.hxx>

#include <TECIO.h>

/**
 * A field is a collection of values for each gridnode.  These values
 * can be accessed by index or index vector, where the index or index
 * vector are valid for the associated LocalGrid.
 *
 * The number of values at each gridnode is returned by getNumElements.  The
 * values returned by getElements are the values in the girdnode of that index.
 */


class NodeFldsBase :  public DynObj
{
 public:
  NodeFldsBase();
  NodeFldsBase(std::string nm, GridGeometry* gridGeom, size_t numComp);
  virtual ~NodeFldsBase();


public:
  void setLengths(Standard_Size lens[3]);

  const ZRGrid* GetZRGrid() const{
    return  GetGridGeom()->GetZRGrid();
  };


  GridGeometry* GetGridGeom() const{
    return m_GridGeom;
  }

  size_t GetElementNum() const;
  size_t getLength(size_t i) const;
  size_t GetSize() const;
  size_t GetSize(size_t i) const;
  
  Standard_Real& operator()(size_t i);
  Standard_Real operator()(size_t i) const;

  Standard_Real& operator()(size_t i, size_t j, size_t dir);
  Standard_Real operator()(size_t i, size_t j, size_t dir) const;


  ////////////////////////////////////////////////////////////////
  //                        Updating
  ////////////////////////////////////////////////////////////////

  virtual void Update();

  /*** Reset all the elements to zero  */
  virtual void Reset();
  
  /**
   * Reset a specific element to zero
   * @param component the component of the field to be reset
   */
  virtual void Reset(size_t component);


  virtual void SetConstComponent(size_t component,Standard_Real constValue);


  virtual void Multiple(Standard_Real theValue); 


  /*** get a pointer to the data for this grid  */
  Standard_Real* GetDataPtr() const;
  
  void Add(NodeFldsBase* theOtherFldsData);
  void Average(NodeFldsBase* theOtherFldsData);
  void Copy(NodeFldsBase* theOtherFldsData);

  virtual void Dump(NodeFlds_OutputBase& dataWriter);

 /***************************************************************/
 
  //void Dump_tecplot(const char * file_name, int step);
  void Dump_tecplot(const char * file_name);
  //void Dump_tecplot(const string& file_prefix , int step);
 void Dump_tecplot_Field_txt(string file_name, int n);	
 
 /***************************************************************/
  

public:
  virtual void SetPhysDataIndexInGridGeom(const Standard_Integer _index);
  virtual void SetupDataSetter();

  virtual void FillWithInterpValue(const vector< TxVector2D<Standard_Real> >& positions,
				   vector< TxVector<Standard_Real> >& values) const;

  virtual void FillWithInterpValue(const TxVector2D<Standard_Real>& position,
				   TxVector<Standard_Real>& value) const;

  virtual void FillWithInterpValue(const vector< TxVector2D<Standard_Real> >& positions,
				   size_t fldComp,
				   vector< Standard_Real >& values ) const;

  virtual void FillWithInterpValue(const IndexAndWeights& indxWt,  
				   TxVector<Standard_Real>& value) const;

  virtual void FillWithInterpValue(const Standard_Size& nb,
				   const vector< IndexAndWeights >& indices,
				   vector< TxVector<Standard_Real> >& values) const;

  virtual void FillWithInterpValue(const TxVector2D<Standard_Real>& position,
				   Standard_Real& result) const;
  
  virtual void FillWithInterpValue(const IndexAndWeights& index,
				   Standard_Real& result) const;

protected:
  GridGeometry* m_GridGeom;

  Standard_Real* m_Data;
  Standard_Size m_DataSize;
  Standard_Size m_Size[3];
  Standard_Size m_Lengths[3];


private:
  // Prevent use
  NodeFldsBase(const NodeFldsBase& vpf){}
  NodeFldsBase& operator=(const NodeFldsBase& vpf){return *this;}


  /**
   * Setup the grid
   * @param numComp the number of components
   */
  void SetupArray(size_t numComp);
  void Setup(std::string nm, GridGeometry* gridGeom, size_t numComp);
};

#endif
