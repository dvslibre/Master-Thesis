function dataCube = readRawDataCubeFromDisk(sTargetLocation, iStartZ, iNumZ, aiDimensions, sFormat, sMachineFormat)
% readRawDataCubeFromDisk
% -----------------------------------------------------------------------
% In:       
%     readRawDataCubeFromDisk(sTargetLocation, iStartZ, iNumZ, aiDimensions, sFormat, sMachineFormat)
%
%     sTargetLocation   (string) filename
%     iStartZ           (interger) first slice to read (defaults to 1 if 0 or negative)
%     iNumZ             (interger) number of slices to read (defaults to maximum if 0 or too large)
%     aiDimensions      (3 intergers) dimensions of data cube
%                                     extracted from filename extension if not specified
%     sFormat           (string) directly passed on to 'fread()', see there
%                       extracted from filename extension if not specified,
%                       defaults to 'float32' of it cannot be extracted
%                       
%                       Here you can also specify the output data format(!) via
%                       "informat=>outformat"
%                       (see see fread(), outformat ALLWAYS defaults to "double" in fread()
%                        so reading an int8 matrix will return a double matrix unless e.g. you
%                        specify 'int8=>int8')
%
%     sMachineFormat    machine format according to fopen()
%                       'b'  -  IEEE float32, big endian
%                       'l'  -  IEEE float32, little endian
%                       's'  -  IEEE float64, big endian
%                       'a'  -  IEEE float64, little endian
%                       'n'  -  native (for all types)
% Out: 
%
% Purpose:    
% 
% See also:   
% 
% Requires: 
%  
% -----------------------------------------------------------------------
% (c) Philips Research Laboratories Aachen, 
%     Molecular Imaging Systems
%     
%     Author:   Rolf Bippus
%     Created:  2004/11/17
%     $Revision: 1.4 $
%     $Date: 2006/05/11 13:37:10 $
%     $Author: Jens Georgi $ (last edit)
% -----------------------------------------------------------------------

    func = 'readRawDataCubeFromDisk';

    % display syntax hint if a wrong number of parameters is given
    if ((nargin<1) | (nargin>6))
        error ('syntax: readRawDataCubeFromDisk(<sTargetLocation>, [ [iNx iNy iNy] , <sFormat>] [, <sMachineFormat>])');
        return;
    end
      
    if (nargin<2)
      iStartZ=0;
    end
    if (nargin<3)
      iNumZ=0;
    end
    if (nargin<4)
       aiDimensions = [0 0 0];
    end
    if (nargin<5)
        sFormat='';
    end
    if (nargin<6)
        sMachineFormat='n';
    end
    
    if iStartZ <= 0
       iStartZ = 1;
    end
    
    iNx = aiDimensions(1);
    iNy = aiDimensions(2);
    iNz = aiDimensions(3);
    
    % regular expresion to match filename in case of missing specs
    regExp = '\S+\.(\d+)x(\d+)x(\d+)\.([^.]+)\.raw';
    tok = regexp(sTargetLocation, regExp, 'tokens'); % 3D case
    if isempty(tok)
        regExp = '\S+\.(\d+)x(\d+)x(\d+)x(\d+)\.([^.]+)\.raw'; % 4D case
        tok = regexp(sTargetLocation, regExp, 'tokens');
    end
      
    if iNx <= 0 | iNy <= 0 | iNz <= 0
            
      if isempty(tok) | size(tok{:}, 2) <= 3 | size(tok{:}, 2) >= 6
        error('%s: missing dimensions specification, cannot get from filename', func);
      end
      if size(tok{:}, 2) == 4
          % C-style sequence -> a[z][y][x] -> nz x ny x nx
          iNz = str2num(tok{1}{1});
          iNy = str2num(tok{1}{2});
          iNx = str2num(tok{1}{3});
      elseif size(tok{:}, 2) == 5
          iNi = str2num(tok{1}{1});
          iNz = str2num(tok{1}{2});
          iNy = str2num(tok{1}{3});
          iNx = str2num(tok{1}{4});
      end
      
    end
    
    if isempty(sFormat)
      if isempty(tok) | size(tok{:}, 2) <= 3 | size(tok{:}, 2) >= 6
        sFormat = 'float32';
      elseif tok{1}{size(tok{:}, 2)}=='Float32'
          sFormat = 'float32';
      else
          sFormat = tok{1}{size(tok{:}, 2)};
      end
    end
    
    % set sFormat if not set externally
    % set sMachineFormat if not set externally

    if sMachineFormat == 'b' | sMachineFormat == 'l'
      if sFormat ~= 'float32'
        error('machine format %s does not match with format %s (see fopen())', sMachineFormat, sFormat);
      end
    end
    if sMachineFormat == 's' | sMachineFormat == 'a'
      if sFormat ~= 'float64'
        error('machine format %s does not match with format %s (see fopen())', sMachineFormat, sFormat);
      end
    end

    if iStartZ < 1
      iStartZ = 1;
    end
    
    if iNumZ <= 0 | iNumZ > (iNz - iStartZ + 1)
      iNumZ = iNz - iStartZ + 1;
    end
    
    % create file handle
    fid=fopen(sTargetLocation, 'r', sMachineFormat);
    
    n = 0;
    if iStartZ > 1
      while (n + iNumZ) < iStartZ
        dataCube = fread (fid, iNx*iNy*iNumZ, sFormat);
        n = n + iNumZ;
      end
      if n < iStartZ - 1;
        dataCube = fread (fid, iNx*iNy*(iStartZ-n-1), sFormat);
        n = iStartZ - 1;
      end
    end    
      
    fprintf('reading %d slices %dx%d format %s starting from slice %d\n', iNumZ, iNx, iNy, sFormat, n+1);
    
    dataCube = fread (fid, iNx*iNy*iNumZ, sFormat);
    size(dataCube);
    dataCube = reshape(dataCube, iNx,iNy,iNumZ);
  
    % % read data from disk (slice-by-slice)
    % % dataCube = zeros(iNx,iNy,iNumZ);
    % for i=1:iNumZ
    %   slice = fread (fid, iNx*iNy, sFormat);
    %   dataCube(:,:,iNumZ+1-i) = flipud(reshape(slice,iNx,iNy));
    % end
    
    % close file handle
    fclose(fid);

% $Log: readRawDataCubeFromDisk.m,v $
% Revision 1.4  2006/05/11 13:37:10  Jens Georgi
% adapted to 4D data
%
% Revision 1.3  2005/02/17 10:55:12  bippus
% introduced extension format for raw data:
%  *.dxdxd.format.raw
%
% Revision 1.2  2004/12/10 15:00:06  bippus
% added machine format (endian)
%
% Revision 1.1  2004/11/17 13:25:41  bippus
% *** empty log message ***
%

