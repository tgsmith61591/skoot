C
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       SUBROUTINE DSWAP(N,DX,INCX,DY,INCY)
* 
*       .. Scalar Arguments ..
*       INTEGER INCX,INCY,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION DX(*),DY(*)
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    interchanges two vectors.
*>    uses unrolled loops for increments equal one.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee 
*> \author Univ. of California Berkeley 
*> \author Univ. of Colorado Denver 
*> \author NAG Ltd. 
*
*> \date November 2011
*
*> \ingroup double_blas_level1
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
C
      subroutine dswap(n,dx,incx,dy,incy)
*
*  -- reference blas level1 routine (version 3.4.0) --
*  -- reference blas is a software package provided by univ. of tennessee,    --
*  -- univ. of california berkeley, univ. of colorado denver and nag ltd..--
*     november 2011
*
*     .. scalar arguments ..
      integer incx,incy,n
*     ..
*     .. array arguments ..
      double precision dx(*),dy(*)
*     ..
*
*  =====================================================================
*
*     .. local scalars ..
      double precision dtemp
      integer i,ix,iy,m,mp1
*     ..
*     .. intrinsic functions ..
      intrinsic mod
*     ..
      if (n.le.0) return
      if (incx.eq.1 .and. incy.eq.1) then
*
*       code for both increments equal to 1
*
*
*       clean-up loop
*
         m = mod(n,3)
         if (m.ne.0) then
            do i = 1,m
               dtemp = dx(i)
               dx(i) = dy(i)
               dy(i) = dtemp
            end do
            if (n.lt.3) return
         end if
         mp1 = m + 1
         do i = mp1,n,3
            dtemp = dx(i)
            dx(i) = dy(i)
            dy(i) = dtemp
            dtemp = dx(i+1)
            dx(i+1) = dy(i+1)
            dy(i+1) = dtemp
            dtemp = dx(i+2)
            dx(i+2) = dy(i+2)
            dy(i+2) = dtemp
         end do
      else
*
*       code for unequal increments or equal increments not equal
*         to 1
*
         ix = 1
         iy = 1
         if (incx.lt.0) ix = (-n+1)*incx + 1
         if (incy.lt.0) iy = (-n+1)*incy + 1
         do i = 1,n
            dtemp = dx(ix)
            dx(ix) = dy(iy)
            dy(iy) = dtemp
            ix = ix + incx
            iy = iy + incy
         end do
      end if
      return
      end
*  =====================================================================
      subroutine dscal(n,da,dx,incx)
*
*  -- reference blas level1 routine (version 3.4.0) --
*  -- reference blas is a software package provided by univ. of tennessee,    --
*  -- univ. of california berkeley, univ. of colorado denver and nag ltd..--
*     november 2011
*
*     .. scalar arguments ..
      double precision da
      integer incx,n
*     ..
*     .. array arguments ..
      double precision dx(*)
*     ..
*
*  =====================================================================
*
*     .. local scalars ..
      integer i,m,mp1,nincx
*     ..
*     .. intrinsic functions ..
      intrinsic mod
*     ..
      if (n.le.0 .or. incx.le.0) return
      if (incx.eq.1) then
*
*        code for increment equal to 1
*
*
*        clean-up loop
*
         m = mod(n,5)
         if (m.ne.0) then
            do i = 1,m
               dx(i) = da*dx(i)
            end do
            if (n.lt.5) return
         end if
         mp1 = m + 1
         do i = mp1,n,5
            dx(i) = da*dx(i)
            dx(i+1) = da*dx(i+1)
            dx(i+2) = da*dx(i+2)
            dx(i+3) = da*dx(i+3)
            dx(i+4) = da*dx(i+4)
         end do
      else
*
*        code for increment not equal to 1
*
         nincx = n*incx
         do i = 1,nincx,incx
            dx(i) = da*dx(i)
         end do
      end if
      return
      end
*  =====================================================================
      double precision function dnrm2(n,x,incx)
*
*  -- reference blas level1 routine (version 3.4.0) --
*  -- reference blas is a software package provided by univ. of tennessee,    --
*  -- univ. of california berkeley, univ. of colorado denver and nag ltd..--
*     november 2011
*
*     .. scalar arguments ..
      integer incx,n
*     ..
*     .. array arguments ..
      double precision x(*)
*     ..
*
*  =====================================================================
*
*     .. parameters ..
      double precision one,zero
      parameter(one=1.0d+0,zero=0.0d+0)
*     ..
*     .. local scalars ..
      double precision absxi,norm,scale,ssq
      integer ix
*     ..
*     .. intrinsic functions ..
      intrinsic abs,sqrt
*     ..
      if (n.lt.1 .or. incx.lt.1) then
          norm = zero
      else if (n.eq.1) then
          norm = abs(x(1))
      else
          scale = zero
          ssq = one
*        the following loop is equivalent to this call to the lapack
*        auxiliary routine:
*        call dlassq( n, x, incx, scale, ssq )
*
          do 10 ix = 1,1 + (n-1)*incx,incx
              if (x(ix).ne.zero) then
                  absxi = abs(x(ix))
                  if (scale.lt.absxi) then
                      ssq = one + ssq* (scale/absxi)**2
                      scale = absxi
                  else
                      ssq = ssq + (absxi/scale)**2
                  end if
              end if
   10     continue
          norm = scale*sqrt(ssq)
      end if
*
      dnrm2 = norm
      return
*
*     end of dnrm2.
*
      end

*  =====================================================================
      double precision function ddot(n,dx,incx,dy,incy)
*
*  -- reference blas level1 routine (version 3.4.0) --
*  -- reference blas is a software package provided by univ. of tennessee,    --
*  -- univ. of california berkeley, univ. of colorado denver and nag ltd..--
*     november 2011
*
*     .. scalar arguments ..
      integer incx,incy,n
*     ..
*     .. array arguments ..
      double precision dx(*),dy(*)
*     ..
*
*  =====================================================================
*
*     .. local scalars ..
      double precision dtemp
      integer i,ix,iy,m,mp1
*     ..
*     .. intrinsic functions ..
      intrinsic mod
*     ..
      ddot = 0.0d0
      dtemp = 0.0d0
      if (n.le.0) return
      if (incx.eq.1 .and. incy.eq.1) then
*
*        code for both increments equal to 1
*
*
*        clean-up loop
*
         m = mod(n,5)
         if (m.ne.0) then
            do i = 1,m
               dtemp = dtemp + dx(i)*dy(i)
            end do
            if (n.lt.5) then
               ddot=dtemp
            return
            end if
         end if
         mp1 = m + 1
         do i = mp1,n,5
          dtemp = dtemp + dx(i)*dy(i) + dx(i+1)*dy(i+1) +
     $            dx(i+2)*dy(i+2) + dx(i+3)*dy(i+3) + dx(i+4)*dy(i+4)
         end do
      else
*
*        code for unequal increments or equal increments
*          not equal to 1
*
         ix = 1
         iy = 1
         if (incx.lt.0) ix = (-n+1)*incx + 1
         if (incy.lt.0) iy = (-n+1)*incy + 1
         do i = 1,n
            dtemp = dtemp + dx(ix)*dy(iy)
            ix = ix + incx
            iy = iy + incy
         end do
      end if
      ddot = dtemp
      return
      end
c  =====================================================================
      subroutine dcopy(n,dx,incx,dy,incy)
c
c  -- reference blas level1 routine (version 3.4.0) --
c  -- reference blas is a software package provided by univ. of tennessee,    --
c  -- univ. of california berkeley, univ. of colorado denver and nag ltd..--
c     november 2011
c
c     .. scalar arguments ..
      integer incx,incy,n
c     ..
c     .. array arguments ..
      double precision dx(*),dy(*)
c     ..
c
c  =====================================================================
c
c     .. local scalars ..
      integer i,ix,iy,m,mp1
c     ..
c     .. intrinsic functions ..
      intrinsic mod
c     ..
      if (n.le.0) return
      if (incx.eq.1 .and. incy.eq.1) then
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
         m = mod(n,7)
         if (m.ne.0) then
            do i = 1,m
               dy(i) = dx(i)
            end do
            if (n.lt.7) return
         end if   
         mp1 = m + 1
         do i = mp1,n,7
            dy(i) = dx(i)
            dy(i+1) = dx(i+1)
            dy(i+2) = dx(i+2)
            dy(i+3) = dx(i+3)
            dy(i+4) = dx(i+4)
            dy(i+5) = dx(i+5)
            dy(i+6) = dx(i+6)
         end do
      else      
c
c        code for unequal increments or equal increments
c          not equal to 1
c
         ix = 1
         iy = 1
         if (incx.lt.0) ix = (-n+1)*incx + 1
         if (incy.lt.0) iy = (-n+1)*incy + 1
         do i = 1,n
            dy(iy) = dx(ix)
            ix = ix + incx
            iy = iy + incy
         end do
      end if
      return
      end
c
      subroutine daxpy(n,da,dx,incx,dy,incy)
c
c  -- reference blas level1 routine (version 3.4.0) --
c  -- reference blas is a software package provided by univ. of tennessee,    --
c  -- univ. of california berkeley, univ. of colorado denver and nag ltd..--
c     november 2011
c
c     .. scalar arguments ..
      double precision da
      integer incx,incy,n
c     ..
c     .. array arguments ..
      double precision dx(*),dy(*)
c     ..
c
c  =====================================================================
c
c     .. local scalars ..
      integer i,ix,iy,m,mp1
c     ..
c     .. intrinsic functions ..
      intrinsic mod
c     ..
      if (n.le.0) return
      if (da.eq.0.0d0) return
      if (incx.eq.1 .and. incy.eq.1) then
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
         m = mod(n,4)
         if (m.ne.0) then
            do i = 1,m
               dy(i) = dy(i) + da*dx(i)
            end do
         end if
         if (n.lt.4) return
         mp1 = m + 1
         do i = mp1,n,4
            dy(i) = dy(i) + da*dx(i)
            dy(i+1) = dy(i+1) + da*dx(i+1)
            dy(i+2) = dy(i+2) + da*dx(i+2)
            dy(i+3) = dy(i+3) + da*dx(i+3)
         end do
      else
c
c        code for unequal increments or equal increments
c          not equal to 1
c
         ix = 1
         iy = 1
         if (incx.lt.0) ix = (-n+1)*incx + 1
         if (incy.lt.0) iy = (-n+1)*incy + 1
         do i = 1,n
          dy(iy) = dy(iy) + da*dx(ix)
          ix = ix + incx
          iy = iy + incy
         end do
      end if
      return
      end
c
c
      subroutine dqrcf(x, n, k, qraux, y, ny, b, info)
      integer, intent(in) :: n
      integer, intent(in) :: k
      integer, intent(in) :: ny
      integer, intent(in) :: info
c
c do we need to make these * instead of specific?
      double precision x(n,*), qraux(*), y(n,*), b(k,*)
c
c declare explicit dependencies
Cf2py intent(inplace) y
Cf2py intent(inplace) b
Cf2py intent(inplace) info
Cf2py depend(n) x
Cf2py depend(n) y
Cf2py depend(k) b
c
c
      integer j
      double precision dummy(1)
      do 10 j = 1,ny
          call dqrsl(x, n, n, k, qraux, y(1,j), dummy,
     &               y(1,j), b(1,j), dummy, dummy, 100, info)
   10 continue
      return
      end
c
c
c     dqrsl applies the output of dqrdc to compute coordinate
c     transformations, projections, and least squares solutions.
c     for k .le. min(n,p), let xk be the matrix
c
c            xk = (x(jpvt(1)),x(jpvt(2)), ... ,x(jpvt(k)))
c
c     formed from columnns jpvt(1), ... ,jpvt(k) of the original
c     n x p matrix x that was input to dqrdc (if no pivoting was
c     done, xk consists of the first k columns of x in their
c     original order).  dqrdc produces a factored orthogonal matrix q
c     and an upper triangular matrix r such that
c
c              xk = q * (r)
c                       (0)
c
c     this information is contained in coded form in the arrays
c     x and qraux.
c
c     on entry
c
c        x      double precision(ldx,p).
c               x contains the output of dqrdc.
c
c        ldx    integer.
c               ldx is the leading dimension of the array x.
c
c        n      integer.
c               n is the number of rows of the matrix xk.  it must
c               have the same value as n in dqrdc.
c
c        k      integer.
c               k is the number of columns of the matrix xk.  k
c               must nnot be greater than min(n,p), where p is the
c               same as in the calling sequence to dqrdc.
c
c        qraux  double precision(p).
c               qraux contains the auxiliary output from dqrdc.
c
c        y      double precision(n)
c               y contains an n-vector that is to be manipulated
c               by dqrsl.
c
c        job    integer.
c               job specifies what is to be computed.  job has
c               the decimal expansion abcde, with the following
c               meaning.
c
c                    if a.ne.0, compute qy.
c                    if b,c,d, or e .ne. 0, compute qty.
c                    if c.ne.0, compute b.
c                    if d.ne.0, compute rsd.
c                    if e.ne.0, compute xb.
c
c               note that a request to compute b, rsd, or xb
c               automatically triggers the computation of qty, for
c               which an array must be provided in the calling
c               sequence.
c
c     on return
c
c        qy     double precision(n).
c               qy conntains q*y, if its computation has been
c               requested.
c
c        qty    double precision(n).
c               qty contains trans(q)*y, if its computation has
c               been requested.  here trans(q) is the
c               transpose of the matrix q.
c
c        b      double precision(k)
c               b contains the solution of the least squares problem
c
c                    minimize norm2(y - xk*b),
c
c               if its computation has been requested.  (note that
c               if pivoting was requested in dqrdc, the j-th
c               component of b will be associated with column jpvt(j)
c               of the original matrix x that was input into dqrdc.)
c
c        rsd    double precision(n).
c               rsd contains the least squares residual y - xk*b,
c               if its computation has been requested.  rsd is
c               also the orthogonal projection of y onto the
c               orthogonal complement of the column space of xk.
c
c        xb     double precision(n).
c               xb contains the least squares approximation xk*b,
c               if its computation has been requested.  xb is also
c               the orthogonal projection of y onto the column space
c               of x.
c
c        info   integer.
c               info is zero unless the computation of b has
c               been requested and r is exactly singular.  in
c               this case, info is the index of the first zero
c               diagonal element of r and b is left unaltered.
c
c     the parameters qy, qty, b, rsd, and xb are not referenced
c     if their computation is not requested and in this case
c     can be replaced by dummy variables in the calling program.
c     to save storage, the user may in some cases use the same
c     array for different parameters in the calling sequence.  a
c     frequently occuring example is when one wishes to compute
c     any of b, rsd, or xb and does not need y or qty.  in this
c     case one may identify y, qty, and one of b, rsd, or xb, while
c     providing separate arrays for anything else that is to be
c     computed.  thus the calling sequence
c
c          call dqrsl(x,ldx,n,k,qraux,y,dum,y,b,y,dum,110,info)
c
c     will result in the computation of b and rsd, with rsd
c     overwriting y.  more generally, each item in the following
c     list contains groups of permissible identifications for
c     a single callinng sequence.
c
c          1. (y,qty,b) (rsd) (xb) (qy)
c
c          2. (y,qty,rsd) (b) (xb) (qy)
c
c          3. (y,qty,xb) (b) (rsd) (qy)
c
c          4. (y,qy) (qty,b) (rsd) (xb)
c
c          5. (y,qy) (qty,rsd) (b) (xb)
c
c          6. (y,qy) (qty,xb) (b) (rsd)
c
c     in any group the value returned in the array allocated to
c     the group corresponds to the last member of the group.
c
c     linpack. this version dated 08/14/78 .
c     g.w. stewart, university of maryland, argonne national lab.
c
c     dqrsl uses the following functions and subprograms.
c
c     BLAS      daxpy,dcopy,ddot
c     Fortran   dabs,min0,mod
c
      subroutine dqrsl(x,ldx,n,k,qraux,y,qy,qty,b,rsd,xb,job,info)
      integer ldx,n,k,job,info
      double precision x(ldx,*),qraux(*),y(*),qy(*),qty(*),b(*),rsd(*),
     *                 xb(*)
c
c declare explicit dependencies
Cf2py intent(in) ldx
Cf2py intent(in) n
Cf2py intent(in) k
Cf2py intent(in) job
Cf2py intent(inplace) b
Cf2py depend(ldx) x
c     internal variables
c
      integer i,j,jj,ju,kp1
      double precision ddot,t,temp
      logical cb,cqy,cqty,cr,cxb
c
c
c     set info flag.
c
      info = 0
c
c     determine what is to be computed.
c
      cqy = job/10000 .ne. 0
      cqty = mod(job,10000) .ne. 0
      cb = mod(job,1000)/100 .ne. 0
      cr = mod(job,100)/10 .ne. 0
      cxb = mod(job,10) .ne. 0
      ju = min0(k,n-1)
c
c     special action when n=1.
c
      if (ju .ne. 0) go to 40
         if (cqy) qy(1) = y(1)
         if (cqty) qty(1) = y(1)
         if (cxb) xb(1) = y(1)
         if (.not.cb) go to 30
            if (x(1,1) .ne. 0.0d0) go to 10
               info = 1
            go to 20
   10       continue
               b(1) = y(1)/x(1,1)
   20       continue
   30    continue
         if (cr) rsd(1) = 0.0d0
      go to 250
   40 continue
c
c        set up to compute qy or qty.
c
         if (cqy) call dcopy(n,y,1,qy,1)
         if (cqty) call dcopy(n,y,1,qty,1)
         if (.not.cqy) go to 70
c
c           compute qy.
c
            do 60 jj = 1, ju
               j = ju - jj + 1
               if (qraux(j) .eq. 0.0d0) go to 50
                  temp = x(j,j)
                  x(j,j) = qraux(j)
                  t = -ddot(n-j+1,x(j,j),1,qy(j),1)/x(j,j)
                  call daxpy(n-j+1,t,x(j,j),1,qy(j),1)
                  x(j,j) = temp
   50          continue
   60       continue
   70    continue
         if (.not.cqty) go to 100
c
c           compute trans(q)*y.
c
            do 90 j = 1, ju
               if (qraux(j) .eq. 0.0d0) go to 80
                  temp = x(j,j)
                  x(j,j) = qraux(j)
                  t = -ddot(n-j+1,x(j,j),1,qty(j),1)/x(j,j)
                  call daxpy(n-j+1,t,x(j,j),1,qty(j),1)
                  x(j,j) = temp
   80          continue
   90       continue
  100    continue
c
c        set up to compute b, rsd, or xb.
c
         if (cb) call dcopy(k,qty,1,b,1)
         kp1 = k + 1
         if (cxb) call dcopy(k,qty,1,xb,1)
         if (cr .and. k .lt. n) call dcopy(n-k,qty(kp1),1,rsd(kp1),1)
         if (.not.cxb .or. kp1 .gt. n) go to 120
            do 110 i = kp1, n
               xb(i) = 0.0d0
  110       continue
  120    continue
         if (.not.cr) go to 140
            do 130 i = 1, k
               rsd(i) = 0.0d0
  130       continue
  140    continue
         if (.not.cb) go to 190
c
c           compute b.
c
            do 170 jj = 1, k
               j = k - jj + 1
               if (x(j,j) .ne. 0.0d0) go to 150
                  info = j
c           ......exit
                  go to 180
  150          continue
               b(j) = b(j)/x(j,j)
               if (j .eq. 1) go to 160
                  t = -b(j)
                  call daxpy(j-1,t,x(1,j),1,b,1)
  160          continue
  170       continue
  180       continue
  190    continue
         if (.not.cr .and. .not.cxb) go to 240
c
c           compute rsd or xb as required.
c
            do 230 jj = 1, ju
               j = ju - jj + 1
               if (qraux(j) .eq. 0.0d0) go to 220
                  temp = x(j,j)
                  x(j,j) = qraux(j)
                  if (.not.cr) go to 200
                     t = -ddot(n-j+1,x(j,j),1,rsd(j),1)/x(j,j)
                     call daxpy(n-j+1,t,x(j,j),1,rsd(j),1)
  200             continue
                  if (.not.cxb) go to 210
                     t = -ddot(n-j+1,x(j,j),1,xb(j),1)/x(j,j)
                     call daxpy(n-j+1,t,x(j,j),1,xb(j),1)
  210             continue
                  x(j,j) = temp
  220          continue
  230       continue
  240    continue
  250 continue
      return
      end
c
      subroutine dqrdc(x,ldx,n,p,qraux,jpvt,work,job)
c
      integer ldx,n,p,job
      integer jpvt(*)
      double precision x(ldx,*),qraux(*),work(*)
c
c     we want to affect these in place...   
Cf2py intent(in) ldx  
Cf2py intent(inplace) x
Cf2py intent(inplace) qraux
Cf2py intent(inplace) jpvt
Cf2py depend(ldx) x
c
c     dqrdc uses householder transformations to compute the qr
c     factorization of an n by p matrix x.  column pivoting
c     based on the 2-norms of the reduced columns may be
c     performed at the users option.
c
c     on entry
c
c        x       double precision(ldx,p), where ldx .ge. n.
c                x contains the matrix whose decomposition is to be
c                computed.
c
c        ldx     integer.
c                ldx is the leading dimension of the array x.
c
c        n       integer.
c                n is the number of rows of the matrix x.
c
c        p       integer.
c                p is the number of columns of the matrix x.
c
c        jpvt    integer(p).
c                jpvt contains integers that control the selection
c                of the pivot columns.  the k-th column x(k) of x
c                is placed in one of three classes according to the
c                value of jpvt(k).
c
c                   if jpvt(k) .gt. 0, then x(k) is an initial
c                                      column.
c
c                   if jpvt(k) .eq. 0, then x(k) is a free column.
c
c                   if jpvt(k) .lt. 0, then x(k) is a final column.
c
c                before the decomposition is computed, initial columns
c                are moved to the beginning of the array x and final
c                columns to the end.  both initial and final columns
c                are frozen in place during the computation and only
c                free columns are moved.  at the k-th stage of the
c                reduction, if x(k) is occupied by a free column
c                it is interchanged with the free column of largest
c                reduced norm.  jpvt is not referenced if
c                job .eq. 0.
c
c        work    double precision(p).
c                work is a work array.  work is not referenced if
c                job .eq. 0.
c
c        job     integer.
c                job is an integer that initiates column pivoting.
c                if job .eq. 0, no pivoting is done.
c                if job .ne. 0, pivoting is done.
c
c     on return
c
c        x       x contains in its upper triangle the upper
c                triangular matrix r of the qr factorization.
c                below its diagonal x contains information from
c                which the orthogonal part of the decomposition
c                can be recovered.  note that if pivoting has
c                been requested, the decomposition is not that
c                of the original matrix x but that of x
c                with its columns permuted as described by jpvt.
c
c        qraux   double precision(p).
c                qraux contains further information required to recover
c                the orthogonal part of the decomposition.
c
c        jpvt    jpvt(k) contains the index of the column of the
c                original matrix that has been interchanged into
c                the k-th column, if pivoting was requested.
c
c     linpack. this version dated 08/14/78 .
c     g.w. stewart, university of maryland, argonne national lab.
c
c     dqrdc uses the following functions and subprograms.
c
c     blas daxpy,ddot,dscal,dswap,dnrm2
c     fortran dabs,dmax1,min0,dsqrt
c
c     internal variables
c
      integer j,jp,l,lp1,lup,maxj,pl,pu
      double precision maxnrm,dnrm2,tt
      double precision ddot,nrmxl,t
      logical negj,swapj
c
c
      pl = 1
      pu = 0
      if (job .eq. 0) go to 60
c
c        pivoting has been requested.  rearrange the columns
c        according to jpvt.
c
         do 20 j = 1, p
            swapj = jpvt(j) .gt. 0
            negj = jpvt(j) .lt. 0
            jpvt(j) = j
            if (negj) jpvt(j) = -j
            if (.not.swapj) go to 10
               if (j .ne. pl) call dswap(n,x(1,pl),1,x(1,j),1)
               jpvt(j) = jpvt(pl)
               jpvt(pl) = j
               pl = pl + 1
   10       continue
   20    continue
         pu = p
         do 50 jj = 1, p
            j = p - jj + 1
            if (jpvt(j) .ge. 0) go to 40
               jpvt(j) = -jpvt(j)
               if (j .eq. pu) go to 30
                  call dswap(n,x(1,pu),1,x(1,j),1)
                  jp = jpvt(pu)
                  jpvt(pu) = jpvt(j)
                  jpvt(j) = jp
   30          continue
               pu = pu - 1
   40       continue
   50    continue
   60 continue
c
c     compute the norms of the free columns.
c
      if (pu .lt. pl) go to 80
      do 70 j = pl, pu
         qraux(j) = dnrm2(n,x(1,j),1)
         work(j) = qraux(j)
   70 continue
   80 continue
c
c     perform the householder reduction of x.
c
      lup = min0(n,p)
      do 200 l = 1, lup
         if (l .lt. pl .or. l .ge. pu) go to 120
c
c           locate the column of largest norm and bring it
c           into the pivot position.
c
            maxnrm = 0.0d0
            maxj = l
            do 100 j = l, pu
               if (qraux(j) .le. maxnrm) go to 90
                  maxnrm = qraux(j)
                  maxj = j
   90          continue
  100       continue
            if (maxj .eq. l) go to 110
               call dswap(n,x(1,l),1,x(1,maxj),1)
               qraux(maxj) = qraux(l)
               work(maxj) = work(l)
               jp = jpvt(maxj)
               jpvt(maxj) = jpvt(l)
               jpvt(l) = jp
  110       continue
  120    continue
         qraux(l) = 0.0d0
         if (l .eq. n) go to 190
c
c           compute the householder transformation for column l.
c
            nrmxl = dnrm2(n-l+1,x(l,l),1)
            if (nrmxl .eq. 0.0d0) go to 180
               if (x(l,l) .ne. 0.0d0) nrmxl = dsign(nrmxl,x(l,l))
               call dscal(n-l+1,1.0d0/nrmxl,x(l,l),1)
               x(l,l) = 1.0d0 + x(l,l)
c
c              apply the transformation to the remaining columns,
c              updating the norms.
c
               lp1 = l + 1
               if (p .lt. lp1) go to 170
               do 160 j = lp1, p
                  t = -ddot(n-l+1,x(l,l),1,x(l,j),1)/x(l,l)
                  call daxpy(n-l+1,t,x(l,l),1,x(l,j),1)
                  if (j .lt. pl .or. j .gt. pu) go to 150
                  if (qraux(j) .eq. 0.0d0) go to 150
                     tt = 1.0d0 - (dabs(x(l,j))/qraux(j))**2
                     tt = dmax1(tt,0.0d0)
                     t = tt
                     tt = 1.0d0 + 0.05d0*tt*(qraux(j)/work(j))**2
                     if (tt .eq. 1.0d0) go to 130
                        qraux(j) = qraux(j)*dsqrt(t)
                     go to 140
  130                continue
                        qraux(j) = dnrm2(n-l,x(l+1,j),1)
                        work(j) = qraux(j)
  140                continue
  150             continue
  160          continue
  170          continue
c
c              save the transformation.
c
               qraux(l) = x(l,l)
               x(l,l) = -nrmxl
  180       continue
  190    continue
  200 continue
      return
      end