
#ifndef CFLDS_SCHEMA
#define CFLDS_SCHEMA

#include <string>

struct CFLDSSchema {

  // Elements of schema
  static std::string typeAtt;
  static std::string kindAtt;
  static std::string meshAtt;

  static std::string nodeOffsetAtt;

  static std::string centeringAtt; // This is deprecated
  static std::string indexOrderAtt; //component major/minor, index C/Fortran; compMinorC is default

  static std::string labelsAtt;
  static std::string axisLabelsAtt;

  static std::string varKey;
  static std::string varsKey;
  static std::string meshKey;

  static std::string nodalCenteringKey;// Default
  static std::string edgeCenteringKey;
  static std::string faceCenteringKey;
  static std::string zonalCenteringKey;

  // Index ordering...
  static std::string compMajorCKey;
  static std::string compMinorCKey;
  static std::string compMajorFKey;
  static std::string compMinorFKey;
  

  struct Uniform {
    static std::string key;
    static std::string lowerBounds;
    static std::string startCell;
    static std::string numCells;
    static std::string upperBounds;
  };

  //time
  static std::string timeKey;
  static std::string timeAtt;
  static std::string cycleAtt;
  
  //run info
  static std::string runInfoKey;
  static std::string softwareAtt;
};

#endif

