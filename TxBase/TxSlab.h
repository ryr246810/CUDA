// ----------------------------------------------------------------------
// File:	TxSlab.h
// Purpose:	Rectanguloid of arbitrary type
// ----------------------------------------------------------------------

#ifndef TX_SLAB_H
#define TX_SLAB_H

// std includes
#include <vector>


// txbase includes
#include <TxStreams.h>
#include <TxAttributeSet.h>

/**
* A grid region is a cubic region of the grid defined by upper and lower bounds for each of the dimensions.  Its information is
* contained in a TxTensor base class that holds the bound information. The TxSlab adds access by bound name (upper, lower, by index)
* and it adds the intersection operator & and the union operator |. The union operator returns the smallest grid region that contains
* both grids.  It also adds the addition and subtraction operators, which shift the first region by the origin of the second.
*/
template<class TYPE>class TxSlab
{
 public:

  TxSlab() {
    for(int i=0; i<3; ++i){
      lowerBnd[i] = 0;
      upperBnd[i] = 0;
    }
    volume = 0;
  };

  /**
   * Construct for given bounds
   * @param lb the new upper bounds
   * @param ub the new upper bounds
   */
 TxSlab(TYPE lb[3], TYPE ub[3]) {
    for(int i=0; i<3; ++i){
      lowerBnd[i] = lb[i];
      upperBnd[i] = ub[i];
    }

    calcVolume();
  }
  
  /*** copy constructor relies on base class, which holds all the data.  */
 TxSlab(const TxSlab<TYPE>& vp) { 
    for(int i=0; i<3; ++i){
      lowerBnd[i] = vp.lowerBnd[i];
      upperBnd[i] = vp.upperBnd[i];
    }
    volume = vp.volume; 
  }
  
  virtual ~TxSlab(){}

  /*** assignment relies on base class, which holds all the data.  */
  TxSlab<TYPE>& operator=(const TxSlab<TYPE>& vp){
    for(int i=0; i<3; ++i){
      lowerBnd[i] = vp.lowerBnd[i];
      upperBnd[i] = vp.upperBnd[i];
    }
    volume = vp.volume;
    return *this;
  }

  //*******************************************************
  //                     Accessors
  //*******************************************************
  
  /**
   * Get the lower bound of a given direction
   * @param i the direction
   * @return the lower bound for that direction
   */
  TYPE getLowerBound(int i) const {
    return lowerBnd[i];
  }
  
  /**
   * Set the lower bound of a given direction
   * @param i the direction
   * @param lb the new lower bound
   */
  void setLowerBound(int i, TYPE lb){ 
    lowerBnd[i] = lb;
    calcVolume();
  }
  /**
   * Get the upper bound of a given direction
   * @param i the direction
   * @return the upper bound for that direction
   */
  TYPE getUpperBound(int i) const {
    return upperBnd[i];
  }
  /**
   * Set the upper bound of a given direction
   * @param i the direction
   * @param lb the new upper bound
   */
  void setUpperBound(int i, TYPE lb){
    upperBnd[i] = lb;
    calcVolume();
  }
  /**
   * Set all bounds at once
   * @param lb the new upper bounds
   * @param ub the new upper bounds
   */
  void setBounds(TYPE lb[3], TYPE ub[3]){
    for(int i=0; i<3; ++i){
      lowerBnd[i] = lb[i];
      upperBnd[i] = ub[i];
    }
    calcVolume();
  }

  /**
   * Set all bounds at once
   * @param lb the new upper bounds
   * @param ub the new upper bounds
   */
  void setBounds(TYPE lb0,TYPE lb1, TYPE lb2,  TYPE ub0, TYPE ub1, TYPE ub2){
    lowerBnd[0] = lb0;
    lowerBnd[1] = lb1;
    lowerBnd[2] = lb2;
    upperBnd[0] = ub0;
    upperBnd[1] = ub1;
    upperBnd[2] = ub2;

    calcVolume();
  }


  /**
   * get the length of this slab in a given direction
   * @param dir the direction
   * @return the length
   */
  TYPE getLength(int dir) const {
    return getUpperBound(dir)-getLowerBound(dir); 
  }


  /**
   * Shift the region 
   * @param i the direction to shift
   * @param a the amount to shift
   */
  void shift(int i, TYPE a){ 
    lowerBnd[i] += a;
    upperBnd[i] += a;
  }


  /**
   * Shift the region 
   * @param a[3] the amount to shift
   */
  void shift(TYPE a[3]){ 
    for(int i=0;i<3;i++){
      lowerBnd[i] += a[i];
      upperBnd[i] += a[i];
    }
  }

  /**
   * Shift the region forward by the origin of another
   * * @param rgn the region to shift by
   */
  void shift(const TxSlab<TYPE>& rgn){ 
    for(int i=0; i<3; ++i){
      lowerBnd[i] += rgn.getLowerBound(i);
      upperBnd[i] += rgn.getLowerBound(i);
    }
  }

  /**
   * Shift the region back by the origin of another This method should not be used, instead use localize of VpDomain or VpDecomp.
   * @param rgn the region to shift by
   */
  void shiftBack(const TxSlab<TYPE>& rgn){ 
    for(int i=0; i<3; ++i) {
      lowerBnd[i] -= rgn.getLowerBound(i);
      upperBnd[i] -= rgn.getLowerBound(i);
    }
  }

  /**
   * Get the volume
   * @return the volume
   */
  TYPE getVolume() const {
    return volume; 
  }

  /**
   * Intersection operator: find largest region in both
   * @param vp the grid region to find union with
   * @return the union of the two regions
   */
  TxSlab<TYPE> operator&(const TxSlab<TYPE>& vp) const{
    TxSlab<TYPE> result;
    for(int i=0; i<3; ++i){
      TYPE b = getLowerBound(i) > vp.getLowerBound(i) ? getLowerBound(i) : vp.getLowerBound(i);
      result.setLowerBound(i,b);
      b = getUpperBound(i) < vp.getUpperBound(i) ? getUpperBound(i) : vp.getUpperBound(i);
      result.setUpperBound(i,b);
    }
    result.calcVolume();
    return result;
  }
  
  /**
   * Union operator: find smallest region containing both
   * @param vp the grid region to find union with
   * @return the union of the two regions
   */
  TxSlab<TYPE> operator|(const TxSlab<TYPE>& vp) const{
    TxSlab<TYPE> result;
    for(int i=0; i<3; ++i){
      TYPE b = getLowerBound(i) < vp.getLowerBound(i) ? getLowerBound(i) : vp.getLowerBound(i);
      result.setLowerBound(i,b);
      b = getUpperBound(i) > vp.getUpperBound(i) ? getUpperBound(i) : vp.getUpperBound(i);
      result.setUpperBound(i,b);
    }
    result.calcVolume();
    return result;
  }

  /*** print out the boundaries in each direction of this slab */
  void writeSlab() const { writeSlab(cout); }

  /*** print out the boundaries in each direction of this slab */
  void writeSlab(ostream& ostr) const{
    ostr << getLowerBound(0) << " - " << getUpperBound(0);
    for(int i=1; i<3; ++i){
      ostr << ", " << getLowerBound(i) << " - " << getUpperBound(i);
    }
    ostr << endl;
  }

  /*** print out the boundaries in each direction of this slab */
  void write() const { writeSlab(); }
  /*** print out the boundaries in each direction of this slab */
  void write(ostream& ostr) const{
    writeSlab(ostr);
  }

  /*** Boolean operator: return whether region is empty **/
  bool isEmpty() const { return !volume; }
  

  /*** Boolean operator: returns, whether region is a subspace */
  bool isSubSpace() const{
    bool subSpace = true;
    for(int i=0; i<3; ++i)
      subSpace = subSpace && (getLowerBound(i) <= getUpperBound(i));
    return subSpace;
  }


  /*** Boolean operator: returns, whether region is a zerospace */
  bool isPureZeroSpace() const{
    bool zeroSpace = true;
    for(int i=0; i<3; ++i)
      zeroSpace = zeroSpace && ( getLowerBound(i) == getUpperBound(i) );
    return zeroSpace;
  }


  bool isDefinedProperly() const{
    bool result = true;
    for(int i=0; i<3; ++i){
      if(getLowerBound(i) > getUpperBound(i)){
	result = false;
	break;
      }
    }
    return result;
  }

  /*** Boolean operator: returns, whether a given position is within the slab or not. Positions on the upper bound are assumed to be outside */
  bool isInside(const TYPE* pos) const {
    bool isInside = true;
    int j = 0;
    while(j < 3 && isInside){
      isInside &= ((pos[j] >= getLowerBound(j)) && (pos[j] < getUpperBound(j)));
      j++;
    }
    return isInside;
  }


  bool isPureInside(const TYPE* pos) const {
    bool isInside = true;
    int j = 0;
    while(j < 3 && isInside){
      isInside &= ((pos[j] > getLowerBound(j)) && (pos[j] < getUpperBound(j)));
      j++;
    }
    return isInside;
  }


 protected:
  /*** Calculate the number of cells **/
  void calcVolume(){
    volume = 1;
    for(int i=0; i<3; ++i){
      if(getLowerBound(i) >= getUpperBound(i)){
	volume = 0;
	break;
      }
      volume *= (getUpperBound(i) - getLowerBound(i));
    }
  }

 private:
  TYPE volume;
  TYPE lowerBnd[3];
  TYPE upperBnd[3];
};

#endif
