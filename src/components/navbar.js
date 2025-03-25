import React from "react"
import styled from "styled-components"
import { Link } from "gatsby"

import { navLinks } from "../../config"

const StyledNav = styled.nav`
  display: none;
  @media (min-width: ${({ theme }) => theme.breakpoints.lg}) {
    display: flex;
    align-items: center;
    background: ${({ theme }) => theme.colors.background};
  }

  .nav-links {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 1.25rem;
    margin-left: auto;
  }

  .nav-link {
    font-size: 1rem;
    font-weight: 700;
    text-align: center;
    position: relative;
    padding: 0;
    color: ${({ theme }) => theme.colors.primary};

    &::before {
      transition: 200ms ease-out;
      height: 0.1563rem;
      content: "";
      position: absolute;
      background-color: ${({ theme }) => theme.colors.primary};
      width: 0%;
      bottom: -0.125rem;
    }
    &:hover::before {
      width: 100%;
    }
  }

  .cta-btn {
    width: auto;
    height: auto;
    font-weight: 700;
    border-radius: ${({ theme }) => theme.borderRadius};
    border: 0.125rem solid ${({ theme }) => theme.colors.primary};
    background: ${({ theme }) => theme.colors.background};
    transition: 200ms ease-out;
    font-size: 1rem;
    padding: 0.5rem 1.5rem;
    margin-left: 1.5rem; /* Ensures spacing between menu and button */

    &:hover {
      background: ${({ theme }) => theme.colors.primary};
      color: ${({ theme }) => theme.colors.background};
    }
  }
`

const Navbar = () => {
  const { menu, button } = navLinks
  return (
    <StyledNav>
      <div className="nav-links">
        {menu.map(({ name, url }, key) => (
          <Link className="nav-link" key={key} to={url}>
            {name}
          </Link>
        ))}
      </div>
      {/* <Link className="cta-btn" to={button.url}>
        {button.name}
      </Link> */}
    </StyledNav>
  )
}

export default Navbar
