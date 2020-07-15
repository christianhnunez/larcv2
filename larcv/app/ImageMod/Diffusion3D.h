/**
 * \file Diffusion3D.h
 *
 * \ingroup ImageMod
 * 
 * \brief Class def header for a class Diffusion3D
 *
 * @author kazuhiro
 */

/** \addtogroup ImageMod

    @{*/
#ifndef __DIFFUSION3D_H__
#define __DIFFUSION3D_H__

#include "larcv/core/Processor/ProcessBase.h"
#include "larcv/core/Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class Diffusion3D ... these comments are used to generate
     doxygen documentation!
  */
  class Diffusion3D : public ProcessBase {

  public:
    
    /// Default constructor
    Diffusion3D(const std::string name="Diffusion3D");
    
    /// Default destructor
    ~Diffusion3D(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  private:

    void configure_labels(const PSet& cfg);
    std::vector<std::string> _output_producer_v;
    std::vector<std::string> _cluster3d_producer_v;
    std::vector<size_t> _numvox_v;
    std::vector<double> _sigma2_v;
    std::vector<std::vector<std::vector<double> > > _scale_vvv;
    bool _normalize;
    float _threshold;
    float _division_threshold;
  };

  /**
     \class larcv::Diffusion3DFactory
     \brief A concrete factory class for larcv::Diffusion3D
  */
  class Diffusion3DProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    Diffusion3DProcessFactory() { ProcessFactory::get().add_factory("Diffusion3D",this); }
    /// dtor
    ~Diffusion3DProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new Diffusion3D(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

